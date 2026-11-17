// Experimental CANN 9.1 ABI: 1024-entry metadata and explicit length tensors.
#include <torch/extension.h>
#include "aclnn_common.h"

at::Tensor li_metadata(
    const at::Tensor& cu, const at::Tensor& lens, const at::Tensor& residual,
    const at::Tensor& cuk, const at::Tensor& qlens, int64_t heads, int64_t topk,
    int64_t max_query, int64_t max_key) {
  auto out = at::empty({1024}, cu.options().dtype(at::kInt));
  std::string qlayout = "TND", klayout = "PA_BBND";
  int64_t one = 1, dim = 128, batch = lens.size(0), mask = 3, ratio = 4;
  ACLNN_CMD(aclnnLightningIndexerV2Metadata, cu, cuk, qlens, lens, residual,
            heads, one, dim, topk, batch, max_query, max_key,
            qlayout, klayout, mask, ratio, out);
  return out;
}

std::tuple<at::Tensor, at::Tensor> li_forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& w,
    const at::Tensor& cu, const at::Tensor& lens, const at::Tensor& residual,
    const at::Tensor& cuk, const at::Tensor& qlens, const at::Tensor& table,
    const at::Tensor& metadata, int64_t topk, int64_t max_query,
    bool with_values) {
  auto indices = at::empty({q.size(0), 1, topk}, q.options().dtype(at::kInt));
  auto values = with_values
      ? at::empty({q.size(0), 1, topk}, q.options().dtype(at::kFloat))
      : at::empty({0}, q.options().dtype(at::kFloat));
  c10::optional<at::Tensor> none;
  std::string qlayout = "TND", klayout = "PA_BBND";
  int64_t mask = 3, ratio = 4, return_value = with_values ? 1 : 0;
  ACLNN_CMD(aclnnLightningIndexerV2, q, k, w, cu, cuk, qlens, lens, residual,
            table, none, metadata, topk, max_query, qlayout, klayout, mask,
            ratio, return_value, indices, values);
  return {indices, values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("metadata", &li_metadata);
  m.def("forward", &li_forward);
}
