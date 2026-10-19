"""Composed full-DFC execution; fused implementations replace WholeMoEProgram."""

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
