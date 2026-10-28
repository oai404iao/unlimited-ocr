// Compiled against the installed Torch headers/libraries; no mangled-symbol binding.
#include <fbgemm/Utils.h>

extern "C" const char* fbgemm_selected_isa() {
    using fbgemm::inst_set_t;
    switch (fbgemm::fbgemmInstructionSet()) {
        case inst_set_t::anyarch: return "anyarch";
        case inst_set_t::avx2: return "avx2";
        case inst_set_t::avx512: return "avx512";
        case inst_set_t::avx512_ymm: return "avx512_ymm";
        case inst_set_t::avx512_vnni: return "avx512_vnni";
        case inst_set_t::avx512_vnni_ymm: return "avx512_vnni_ymm";
        default: return "other";
    }
}
