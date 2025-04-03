import unittest
import torch

from sglang.srt.layers.moe.ep_moe.layer import (
    run_moe_ep_preproess_native,
    run_moe_ep_preproess,
    pre_reorder_native,
    pre_reorder_triton_kernel,
    GroupedGemmRunner,
    grouped_gemm_runner_native,
    silu_and_mul_triton_kernel,
    silu_and_mul_native,
    gelu_and_mul_triton_kernel,
    gelu_and_mul_native,
    post_reorder_triton_kernel,
    post_reorder_native,
)


class TestRunMoeEpPreprocessNative(unittest.TestCase):
    def test_random_input(self):
        num_experts = 8
        topk_ids = torch.tensor([[2, 0], [1, 2], [0, 1], [1, 2]]).to("cuda:0")

        expected_reorder_topk_ids, excepted_src2dst, expected_seg_indptr = (
            run_moe_ep_preproess(topk_ids, num_experts)
        )

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess_native(
            topk_ids, num_experts
        )

        self.assertTrue(
            torch.equal(reorder_topk_ids, expected_reorder_topk_ids),
            f"Expected reorder_topk_ids: {expected_reorder_topk_ids}, got: {reorder_topk_ids}",
        )
        self.assertTrue(
            torch.equal(src2dst, excepted_src2dst),
            f"Expected src2dst: {excepted_src2dst}, got: {src2dst}",
        )
        self.assertTrue(
            torch.equal(seg_indptr, expected_seg_indptr),
            f"Expected seg_indptr: {expected_seg_indptr}, got: {seg_indptr}",
        )


class TestPreReorderNative(unittest.TestCase):
    def test_random_input(self):
        token_num = 4
        hidden_size = 16
        num_experts = 8
        num_experts_per_partition = 2
        top_k = 2
        start_expert_id = 5
        end_expert_id = 6
        active_expert_ids = [5, 6]

        hidden_states = torch.randn(token_num, hidden_size).to("cuda:0")
        router_logits = torch.randn(token_num, num_experts).to("cuda:0")
        topk_weights = torch.randn(token_num, top_k).to("cuda:0")
        topk_ids = torch.randint(0, num_experts, (token_num, top_k)).to(
            "cuda:0"
        )

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess_native(
            topk_ids, num_experts
        )

        expected_gateup_input = torch.zeros(
            (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w13_input_scale = max_value / torch.finfo(torch.float8_e4m3fn).max

        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            expected_gateup_input,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        gateup_input = pre_reorder_native(
            hidden_states,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            top_k,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )

        # print(
        #     f"Expected gateup_input: {expected_gateup_input}, got: {gateup_input}"
        # )
        torch.testing.assert_close(
            gateup_input, expected_gateup_input, rtol=1e-4, atol=1e-4
        )


class TestSiluAndMulNative(unittest.TestCase):
    def test_random_input(self):
        token_num = 4
        hidden_size = 16
        num_experts = 8
        num_experts_per_partition = 2
        top_k = 2
        start_expert_id = 5
        end_expert_id = 6
        active_expert_ids = [5, 6]
        intermediate_size = 16

        hidden_states = torch.randn(token_num, hidden_size).to("cuda:0")
        router_logits = torch.randn(token_num, num_experts).to("cuda:0")
        topk_weights = torch.randn(token_num, top_k).to("cuda:0")
        topk_ids = torch.randint(0, num_experts, (token_num, top_k)).to(
            "cuda:0"
        )
        w13_weight = torch.randn(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
        ).to("cuda:0")

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, num_experts
        )

        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w13_input_scale = max_value / torch.finfo(torch.float8_e4m3fn).max

        gateup_input = pre_reorder_native(
            hidden_states,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            top_k,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )

        seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
        seg_ind_interval = [
            (seg_indptr[expert_id], seg_indptr[expert_id + 1])
            for expert_id in active_expert_ids
        ]
        weight_indices_cur_rank = torch.arange(
            0,
            num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        gateup_output = torch.empty(
            gateup_input.shape[0],
            w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_runner = GroupedGemmRunner(
            hidden_states.device,
            use_flashinfer=False,
        )
        w13_weight_scale = torch.ones(
            num_experts_per_partition, dtype=torch.float32
        ).to("cuda:0")

        gateup_output = grouped_gemm_runner(
            a=gateup_input,
            b=w13_weight,
            c=gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w13_input_scale,
            scale_b=w13_weight_scale,
            block_shape=None,
        )

        expected_down_input = torch.zeros(
            (gateup_output.shape[0], gateup_output.shape[1] // 2),
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            expected_down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            BLOCK_SIZE=512,
        )

        down_input = silu_and_mul_native(
            gateup_output,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            dtype=hidden_states.dtype,
        )

        # print(f"Expected down_input: {expected_down_input}, got: {down_input}")
        torch.testing.assert_close(
            down_input, expected_down_input, rtol=1e-4, atol=1e-4
        )


class TestGeluAndMulNative(unittest.TestCase):
    def test_random_input(self):
        token_num = 4
        hidden_size = 16
        num_experts = 8
        num_experts_per_partition = 2
        top_k = 2
        start_expert_id = 5
        end_expert_id = 6
        active_expert_ids = [5, 6]
        intermediate_size = 16

        hidden_states = torch.randn(token_num, hidden_size).to("cuda:0")
        router_logits = torch.randn(token_num, num_experts).to("cuda:0")
        topk_weights = torch.randn(token_num, top_k).to("cuda:0")
        topk_ids = torch.randint(0, num_experts, (token_num, top_k)).to(
            "cuda:0"
        )
        w13_weight = torch.randn(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
        ).to("cuda:0")

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, num_experts
        )

        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w13_input_scale = max_value / torch.finfo(torch.float8_e4m3fn).max

        gateup_input = pre_reorder_native(
            hidden_states,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            top_k,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )

        seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
        seg_ind_interval = [
            (seg_indptr[expert_id], seg_indptr[expert_id + 1])
            for expert_id in active_expert_ids
        ]
        weight_indices_cur_rank = torch.arange(
            0,
            num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        gateup_output = torch.empty(
            gateup_input.shape[0],
            w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_runner = GroupedGemmRunner(
            hidden_states.device,
            use_flashinfer=False,
        )
        w13_weight_scale = torch.ones(
            num_experts_per_partition, dtype=torch.float32
        ).to("cuda:0")

        gateup_output = grouped_gemm_runner(
            a=gateup_input,
            b=w13_weight,
            c=gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w13_input_scale,
            scale_b=w13_weight_scale,
            block_shape=None,
        )

        expected_down_input = torch.zeros(
            (gateup_output.shape[0], gateup_output.shape[1] // 2),
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            expected_down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            BLOCK_SIZE=512,
        )

        down_input = gelu_and_mul_native(
            gateup_output,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            dtype=hidden_states.dtype,
        )

        # print(f"Expected down_input: {expected_down_input}, got: {down_input}")
        torch.testing.assert_close(
            down_input, expected_down_input, rtol=1e-4, atol=1e-4
        )


class TestPostReorderNative(unittest.TestCase):
    def test_random_input(self):
        token_num = 4
        hidden_size = 16
        num_experts = 8
        num_experts_per_partition = 2
        top_k = 2
        start_expert_id = 5
        end_expert_id = 6
        active_expert_ids = [5, 6]
        intermediate_size = 16

        hidden_states = torch.randn(token_num, hidden_size).to("cuda:0")
        router_logits = torch.randn(token_num, num_experts).to("cuda:0")
        topk_weights = torch.randn(token_num, top_k).to("cuda:0")
        topk_ids = torch.randint(0, num_experts, (token_num, top_k)).to(
            "cuda:0"
        )
        w13_weight = torch.randn(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
        ).to("cuda:0")

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, num_experts
        )

        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w13_input_scale = max_value / torch.finfo(torch.float8_e4m3fn).max

        gateup_input = pre_reorder_native(
            hidden_states,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            top_k,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )

        seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
        seg_ind_interval = [
            (seg_indptr[expert_id], seg_indptr[expert_id + 1])
            for expert_id in active_expert_ids
        ]
        weight_indices_cur_rank = torch.arange(
            0,
            num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        gateup_output = torch.empty(
            gateup_input.shape[0],
            w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_runner = GroupedGemmRunner(
            hidden_states.device,
            use_flashinfer=False,
        )
        w13_weight_scale = torch.ones(
            num_experts_per_partition, dtype=torch.float32
        ).to("cuda:0")

        gateup_output = grouped_gemm_runner(
            a=gateup_input,
            b=w13_weight,
            c=gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w13_input_scale,
            scale_b=w13_weight_scale,
            block_shape=None,
        )

        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        down_input = silu_and_mul_native(
            gateup_output,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            hidden_states.dtype,
        )

        w2_weight = torch.randn(
            (num_experts_per_partition,
            hidden_size,
            intermediate_size),
            dtype=hidden_states.dtype,
        ).to("cuda:0")
        w2_weight_scale = torch.ones(
            num_experts_per_partition, dtype=torch.float32
        ).to("cuda:0")

        down_output = torch.empty(
            (down_input.shape[0],
            w2_weight.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        down_output = grouped_gemm_runner(
            a=down_input,
            b=w2_weight,
            c=down_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w2_input_scale,
            scale_b=w2_weight_scale,
            block_shape=None,
        )

        expected_output = torch.empty_like(hidden_states)
        post_reorder_triton_kernel[(hidden_states.size(0),)](
            down_output,
            expected_output,
            src2dst,
            topk_ids,
            topk_weights,
            start_expert_id,
            end_expert_id,
            top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )
        output = post_reorder_native(down_output, src2dst, topk_ids, topk_weights, start_expert_id, end_expert_id, active_expert_ids)

        # print(f"Expected output: {expected_output}, got: {output}")
        torch.testing.assert_close(
            output, expected_output, rtol=1e-4, atol=1e-4
        )

class TestGroupedGemmRunnerNative(unittest.TestCase):
    def test_random_input(self):
        token_num = 4
        hidden_size = 16
        num_experts = 8
        num_experts_per_partition = 2
        top_k = 2
        start_expert_id = 5
        end_expert_id = 6
        active_expert_ids = [5, 6]
        intermediate_size = 16

        hidden_states = torch.randn(token_num, hidden_size).to("cuda:0")
        router_logits = torch.randn(token_num, num_experts).to("cuda:0")
        topk_weights = torch.randn(token_num, top_k).to("cuda:0")
        topk_ids = torch.randint(0, num_experts, (token_num, top_k)).to(
            "cuda:0"
        )
        w13_weight = torch.randn(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
        ).to("cuda:0")

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, num_experts
        )

        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w13_input_scale = max_value / torch.finfo(torch.float8_e4m3fn).max

        gateup_input = pre_reorder_native(
            hidden_states,
            src2dst,
            topk_ids,
            w13_input_scale,
            start_expert_id,
            end_expert_id,
            active_expert_ids,
            top_k,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
        )

        seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
        seg_ind_interval = [
            (seg_indptr[expert_id], seg_indptr[expert_id + 1])
            for expert_id in active_expert_ids
        ]
        weight_indices_cur_rank = torch.arange(
            0,
            num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        expected_gateup_output = torch.empty(
            gateup_input.shape[0],
            w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grouped_gemm_runner = GroupedGemmRunner(
            hidden_states.device,
            use_flashinfer=False,
        )
        w13_weight_scale = torch.ones(
            num_experts_per_partition, dtype=torch.float32
        ).to("cuda:0")

        expected_gateup_output = grouped_gemm_runner(
            a=gateup_input,
            b=w13_weight,
            c=expected_gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w13_input_scale,
            scale_b=w13_weight_scale,
            block_shape=None,
        )

        gateup_input = grouped_gemm_runner_native(
            a=gateup_input,
            b=w13_weight,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_ind_interval=seg_ind_interval,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=w13_input_scale,
            scale_b=w13_weight_scale,
        )

if __name__ == "__main__":
    unittest.main()
