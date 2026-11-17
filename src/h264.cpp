// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include <limits>

namespace irisva {
namespace {
class Bits {
    std::vector<uint8_t> bytes_;
    unsigned position_ = 0;

  public:
    void bit(unsigned value) {
        if (!(position_ & 7))
            bytes_.push_back(0);
        bytes_.back() |= (value & 1) << (7 - (position_++ & 7));
    }
    void bits(uint32_t value, unsigned count) {
        for (unsigned i = count; i; --i)
            bit(value >> (i - 1));
    }
    void ue(uint32_t value) {
        check(value != UINT32_MAX, "Exp-Golomb overflow");
        unsigned n = 0;
        uint32_t code = value + 1;
        for (uint32_t v = code; v >>= 1;)
            ++n;
        for (unsigned i = 0; i < n; ++i)
            bit(0);
        bits(code, n + 1);
    }
    void se(int value) {
        ue(value <= 0 ? -2 * value : 2 * value - 1);
    }
    unsigned position() const {
        return position_;
    }
    void nal(std::vector<uint8_t> &out, uint8_t header, bool trailing = true) {
        if (trailing)
            bit(1);
        while (position_ & 7)
            bit(0);
        out.insert(out.end(), {0, 0, 0, 1, header});
        unsigned zeros = 0;
        for (uint8_t v : bytes_) {
            if (zeros == 2 && v <= 3) {
                out.push_back(3);
                zeros = 0;
            }
            out.push_back(v);
            zeros = v == 0 ? zeros + 1 : 0;
        }
    }
};
class Reader {
    std::vector<uint8_t> bytes_;
    size_t position_ = 0;

  public:
    explicit Reader(const std::vector<uint8_t> &bytes) {
        unsigned zeros = 0;
        for (size_t i = 1; i < bytes.size(); ++i) {
            uint8_t v = bytes[i];
            if (zeros == 2 && v == 3) {
                zeros = 0;
                continue;
            }
            bytes_.push_back(v);
            zeros = v == 0 ? zeros + 1 : 0;
        }
    }
    unsigned bit() {
        check(position_ < bytes_.size() * 8, "truncated H264 slice header",
              VA_STATUS_ERROR_INVALID_BUFFER);
        unsigned p = position_++;
        return (bytes_[p / 8] >> (7 - p % 8)) & 1;
    }
    unsigned position() const {
        return position_;
    }
    unsigned size_bits() const {
        return bytes_.size() * 8;
    }
    void seek(unsigned position) {
        check(position <= size_bits(), "H264 bit position out of bounds");
        position_ = position;
    }
    uint32_t bits(unsigned count) {
        check(count <= 32, "invalid H264 bit field");
        uint32_t v = 0;
        for (unsigned i = 0; i < count; ++i)
            v = (v << 1) | bit();
        return v;
    }
    void copy(Bits &out, unsigned end) {
        check(end >= position_ && end <= size_bits(), "invalid H264 header offset",
              VA_STATUS_ERROR_INVALID_BUFFER);
        while (position_ < end)
            out.bit(bit());
    }
    uint32_t ue() {
        unsigned n = 0;
        while (!bit())
            check(++n < 31, "invalid H264 Exp-Golomb");
        uint32_t v = 1;
        for (unsigned i = 0; i < n; ++i)
            v = (v << 1) | bit();
        return v - 1;
    }
};
// POC type 1's SPS offset cycle is absent from VA-API. Encode the resolved
// picture order counts as type 0 instead, preserving every slice-data bit.
// CABAC data stays byte aligned; CAVLC's rbsp_trailing_bits are regenerated.
void rewrite_poc(const Slice &slice, const VAPictureParameterBufferH264 &p,
                 std::vector<uint8_t> &out) {
    Reader r(slice.bytes);
    r.ue();
    r.ue();
    r.ue();
    r.bits(p.seq_fields.bits.log2_max_frame_num_minus4 + 4);
    bool field = false, bottom = false;
    if (!p.seq_fields.bits.frame_mbs_only_flag) {
        field = r.bit();
        if (field)
            bottom = r.bit();
    }
    if ((slice.bytes[0] & 31) == 5)
        r.ue();
    unsigned poc_start = r.position();
    if (!p.seq_fields.bits.delta_pic_order_always_zero_flag) {
        r.ue(); // signed Exp-Golomb occupies the same bits as unsigned
        if (p.pic_fields.bits.pic_order_present_flag && !field)
            r.ue();
    }
    unsigned poc_end = r.position();
    Bits rewritten;
    r.seek(0);
    r.copy(rewritten, poc_start);
    int poc = bottom ? p.CurrPic.BottomFieldOrderCnt : p.CurrPic.TopFieldOrderCnt;
    rewritten.bits(uint32_t(poc) & 65535, 16);
    if (!field) {
        int64_t delta = int64_t(p.CurrPic.BottomFieldOrderCnt) - p.CurrPic.TopFieldOrderCnt;
        check(delta > INT32_MIN / 2 && delta < INT32_MAX / 2, "H264 field POC delta out of range");
        rewritten.se(int(delta));
    }
    r.seek(poc_end);
    if (p.pic_fields.bits.entropy_coding_mode_flag) {
        check(slice.params.slice_data_bit_offset >= 8, "missing H264 slice header offset",
              VA_STATUS_ERROR_INVALID_BUFFER);
        unsigned end = slice.params.slice_data_bit_offset - 8;
        r.copy(rewritten, end);
        while (rewritten.position() & 7)
            rewritten.bit(1);
        r.seek((end + 7) & ~7u);
        r.copy(rewritten, r.size_bits());
        rewritten.nal(out, slice.bytes[0], false);
    } else {
        unsigned end = r.size_bits();
        // Locate rbsp_stop_one_bit. Trailing zero bytes are not slice data.
        do {
            check(end > poc_end, "H264 missing RBSP stop bit");
            r.seek(--end);
        } while (!r.bit());
        r.seek(poc_end);
        r.copy(rewritten, end);
        rewritten.nal(out, slice.bytes[0]);
    }
}
const unsigned zigzag4[] = {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};
const unsigned zigzag8[] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
void matrix(Bits &b, const uint8_t *values, const unsigned *scan, unsigned n) {
    b.bit(1);
    int last = 8;
    for (unsigned j = 0; j < n; ++j) {
        int value = values[scan[j]];
        check(value != 0, "invalid zero H264 scaling list entry", VA_STATUS_ERROR_INVALID_BUFFER);
        int delta = ((value - last + 128) & 255) - 128;
        b.se(delta);
        last = value;
    }
}
} // namespace

std::vector<uint8_t> h264_bitstream(VAProfile profile, const Picture &pic, H264State &state) {
    check(pic.has_params && !pic.slices.empty() && pic.pending_slices.empty(),
          "H264 picture missing parameters or slices", VA_STATUS_ERROR_INVALID_BUFFER);
    if (!state.started) {
        // A context created mid-GOP can receive dependent pictures before an
        // IDR. Iris consumes them without producing CAPTURE buffers, eventually
        // exhausting the surface pool. Reject them until random access instead.
        bool idr = false;
        for (const auto &slice : pic.slices)
            idr |= !slice.bytes.empty() && (slice.bytes[0] & 31) == 5;
        check(idr, "H264 context requires an initial IDR picture",
              VA_STATUS_ERROR_DECODING_ERROR);
    }
    const auto &p = pic.params;
    const auto &s = p.seq_fields.bits;
    const auto &f = p.pic_fields.bits;
    check(s.chroma_format_idc == 1 && !p.bit_depth_luma_minus8 && !p.bit_depth_chroma_minus8,
          "only H264 8-bit 4:2:0 is supported", VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT);
    check(s.frame_mbs_only_flag, "Iris currently rejects interlaced H264 sequences",
          VA_STATUS_ERROR_UNIMPLEMENTED);
    check(s.log2_max_frame_num_minus4 <= 12 && s.pic_order_cnt_type <= 2 &&
              (s.pic_order_cnt_type != 0 || s.log2_max_pic_order_cnt_lsb_minus4 <= 12) &&
              p.num_ref_frames <= 16,
          "invalid H264 sequence parameters", VA_STATUS_ERROR_INVALID_BUFFER);
    const bool rewrite = s.pic_order_cnt_type == 1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    check(!p.num_slice_groups_minus1, "H264 slice groups unsupported",
          VA_STATUS_ERROR_UNIMPLEMENTED);
#pragma GCC diagnostic pop
    unsigned profile_idc = profile == VAProfileH264High   ? 100
                           : profile == VAProfileH264Main ? 77
                                                          : 66;
    std::vector<uint8_t> out;
    Bits sps;
    sps.bits(profile_idc, 8);
    sps.bits(profile_idc == 66 ? 0xc0 : 0, 8);
    sps.bits(51, 8); // A conservative decoding level, independent of absent original level_idc.
    sps.ue(0);
    if (profile_idc >= 100) {
        sps.ue(1);
        sps.ue(0);
        sps.ue(0);
        sps.bit(0);
        sps.bit(0);
    }
    sps.ue(s.log2_max_frame_num_minus4);
    sps.ue(rewrite ? 0 : s.pic_order_cnt_type);
    if (rewrite || !s.pic_order_cnt_type)
        sps.ue(rewrite ? 12 : s.log2_max_pic_order_cnt_lsb_minus4);
    sps.ue(p.num_ref_frames);
    sps.bit(s.gaps_in_frame_num_value_allowed_flag);
    sps.ue(p.picture_width_in_mbs_minus1);
    // VA height is expressed in frame macroblocks, SPS in map units.
    sps.ue((p.picture_height_in_mbs_minus1 + 1) / (2 - s.frame_mbs_only_flag) - 1);
    sps.bit(s.frame_mbs_only_flag);
    if (!s.frame_mbs_only_flag)
        sps.bit(s.mb_adaptive_frame_field_flag);
    sps.bit(s.direct_8x8_inference_flag);
    sps.bit(0); // VA owns presentation cropping; coded pixels are retained.
    sps.bit(0); // VUI is absent from VA parameters; use the standard inferred DPB limit.
    std::vector<uint8_t> new_sps;
    sps.nal(new_sps, 0x67);
    if (new_sps != state.sps) {
        trace("H264 SPS profile=%u original-poc=%u rewritten-poc=%u", profile_idc,
              s.pic_order_cnt_type, rewrite ? 0 : s.pic_order_cnt_type);
        out.insert(out.end(), new_sps.begin(), new_sps.end());
        state.sps = new_sps;
    }

    std::map<unsigned, std::pair<int, int>> pps_defaults;
    for (const auto &slice : pic.slices) {
        check(!slice.bytes.empty() && ((slice.bytes[0] & 31) == 1 || (slice.bytes[0] & 31) == 5),
              "invalid H264 slice NAL", VA_STATUS_ERROR_INVALID_BUFFER);
        Reader r(slice.bytes);
        r.ue();
        unsigned raw_type = r.ue();
        unsigned id = r.ue();
        check(raw_type <= 9, "invalid H264 slice type", VA_STATUS_ERROR_INVALID_BUFFER);
        unsigned type = raw_type % 5;
        check(type <= 2, "H264 SP/SI switching slices unsupported", VA_STATUS_ERROR_UNIMPLEMENTED);
        check(id <= 255, "invalid H264 PPS ID");
        auto [it, inserted] = pps_defaults.try_emplace(id, -1, -1);
        (void)inserted;
        if (type == 2 || type == 4)
            continue; // I/SI provides no reference defaults.
        r.bits(s.log2_max_frame_num_minus4 + 4);
        bool field = false;
        if (!s.frame_mbs_only_flag) {
            field = r.bit();
            if (field)
                r.bit();
        }
        if ((slice.bytes[0] & 31) == 5)
            r.ue();
        if (!s.pic_order_cnt_type) {
            r.bits(s.log2_max_pic_order_cnt_lsb_minus4 + 4);
            if (f.pic_order_present_flag && !field)
                r.ue();
        } else if (s.pic_order_cnt_type == 1 && !s.delta_pic_order_always_zero_flag) {
            r.ue();
            if (f.pic_order_present_flag && !field)
                r.ue();
        }
        if (f.redundant_pic_cnt_present_flag)
            r.ue();
        if (type == 1)
            r.bit(); // direct_spatial_mv_pred_flag
        if (r.bit())
            continue; // Explicit slice overrides must not overwrite PPS defaults.
        auto merge = [](int &previous, unsigned count) {
            check(count <= 31 && (previous < 0 || unsigned(previous) == count),
                  "inconsistent H264 PPS reference defaults", VA_STATUS_ERROR_INVALID_BUFFER);
            previous = count;
        };
        merge(it->second.first, slice.params.num_ref_idx_l0_active_minus1);
        if (type == 1)
            merge(it->second.second, slice.params.num_ref_idx_l1_active_minus1);
    }
    for (const auto &[id, counts] : pps_defaults) {
        Bits pps;
        pps.ue(id);
        pps.ue(0);
        pps.bit(f.entropy_coding_mode_flag);
        pps.bit(rewrite || f.pic_order_present_flag);
        pps.ue(0);
        pps.ue(std::max(0, counts.first));
        pps.ue(std::max(0, counts.second));
        pps.bit(f.weighted_pred_flag);
        pps.bits(f.weighted_bipred_idc, 2);
        pps.se(p.pic_init_qp_minus26);
        pps.se(p.pic_init_qs_minus26);
        pps.se(p.chroma_qp_index_offset);
        pps.bit(f.deblocking_filter_control_present_flag);
        pps.bit(f.constrained_intra_pred_flag);
        pps.bit(f.redundant_pic_cnt_present_flag);
        if (profile_idc >= 100) {
            pps.bit(f.transform_8x8_mode_flag);
            pps.bit(pic.has_iq);
            if (pic.has_iq) {
                for (unsigned j = 0; j < 6; ++j)
                    matrix(pps, pic.iq.ScalingList4x4[j], zigzag4, 16);
                if (f.transform_8x8_mode_flag)
                    for (unsigned j = 0; j < 2; ++j)
                        matrix(pps, pic.iq.ScalingList8x8[j], zigzag8, 64);
            }
            pps.se(p.second_chroma_qp_index_offset);
        }
        std::vector<uint8_t> new_pps;
        pps.nal(new_pps, 0x68);
        if (new_pps != state.pps[id]) {
            out.insert(out.end(), new_pps.begin(), new_pps.end());
            state.pps[id] = new_pps;
        }
    }
    for (const auto &slice : pic.slices) {
        if (rewrite) {
            rewrite_poc(slice, p, out);
            continue;
        }
        out.insert(out.end(), {0, 0, 0, 1});
        out.insert(out.end(), slice.bytes.begin(), slice.bytes.end());
    }
    state.started = true;
    return out;
}
} // namespace irisva
