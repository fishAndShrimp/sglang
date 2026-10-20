"""DFC schedules independent of weight transport and accelerator layouts."""

from .interfaces import WholeMoEProgram


class ComposedDFCProgram(WholeMoEProgram):
    def __init__(self, waves, dispatcher, local_program):
        self.waves = tuple(waves)
        if not self.waves:
            raise ValueError("A DFC program requires a static nonempty wave schedule")
        self.dispatcher = dispatcher
        self.local_program = local_program

    def run(self, layer, hidden_states, routing, context):
        # This static Python loop is unrolled at capture. Its length never depends
        # on misses or device route contents. A dispatcher handle lives for D/F/C.
        result = self.dispatcher.accumulator(hidden_states, context)
        for wave in self.waves:
            masked = self.dispatcher.mask_routes(routing, wave, context)
            dispatched = self.dispatcher.dispatch(hidden_states, masked, context)
            local = self.local_program.run(layer, dispatched, wave, context)
            partial = self.dispatcher.combine(local, context)
            self.dispatcher.accumulate(result, partial, context)
        return self.dispatcher.finish(result, context)


class DispatchOnceProgram(WholeMoEProgram):
    """One full dispatch, fixed local F waves, then one full combine.

    Bind one instance to a local expert count. The layout adapter supplies views
    or explicitly budgeted packing buffers; the local program owns weight
    readiness and reader protection. Neither transport nor cache policy is
    selected here. This sequential schedule does not itself add H2D overlap.
    """

    def __init__(
        self, waves, dispatcher, local_program, layout_adapter, *, local_experts
    ):
        self.waves = tuple(waves)
        if local_experts <= 0 or not self.waves:
            raise ValueError("Dispatch-once requires a nonempty local expert schedule")
        end = 0
        for index, wave in enumerate(self.waves):
            if (
                wave.index != index
                or wave.local_start != end
                or not end < wave.local_end <= local_experts
                or wave.bank < 0
            ):
                raise ValueError("Waves must partition local experts in order")
            end = wave.local_end
        if end != local_experts:
            raise ValueError("Waves must cover every local expert before dispatch")
        # Capacity/format compatibility of each bank is validated by the runtime.
        self.dispatcher = dispatcher
        self.local_program = local_program
        self.layout_adapter = layout_adapter

    def run(self, layer, hidden_states, routing, context):
        # Preserve original global IDs/weights and keep this D handle through C.
        dispatched = self.dispatcher.dispatch(hidden_states, routing, context)
        layout = self.layout_adapter.bind(dispatched, context)
        # Unrolled at capture, including empty waves. Dynamic rows stay on device.
        for wave in self.waves:
            wave_input = self.layout_adapter.select_wave(layout, wave, context)
            local = self.local_program.run(layer, wave_input, wave, context)
            self.layout_adapter.write_wave(layout, wave, local, context)
        combined = self.dispatcher.combine(
            self.layout_adapter.combine_input(layout, context), context
        )
        return self.dispatcher.finish(combined, context)
