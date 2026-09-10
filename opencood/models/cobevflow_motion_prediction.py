import torch

from opencood.tools.traj_prediction import make_model


class CobevflowMotionPrediction(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.get('input_dim', 4)
        output_dim = args.get('output_dim', 2)
        num_layers = args.get('num_layers', 2)
        d_model = args.get('d_model', 64)
        d_ff = args.get('d_ff', 128)
        num_heads = args.get('num_heads', 2)
        dropout = args.get('dropout', 0.1)
        motion_model, _ = make_model(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            dropout=dropout)

        self.encoder = motion_model.encoder
        self.decoder = motion_model.decoder
        self.src_embed = motion_model.src_embed
        self.tgt_embed = motion_model.tgt_embed
        self.src_pe = motion_model.src_pe
        self.tgt_pe = motion_model.tgt_pe
        self.generator = motion_model.generator

    def forward(self, data_dict, *args, **kwargs):
        obj_input, query, time, future_time, gt_offset = \
            self._build_motion_inputs(data_dict)

        src = self.src_embed(obj_input)
        src = self.src_pe(src, time)
        tgt = self.tgt_embed(query)
        tgt = self.tgt_pe(tgt, future_time)

        memory = self.encoder(src, None)
        pred_residual = self.decoder(tgt, memory, memory, None)
        pred_offset = self.generator(pred_residual) + query
        return {
            'preds_coop': pred_offset.squeeze(0).squeeze(1),
            'gt_coop': gt_offset,
        }

    @staticmethod
    def _expand_cav_times(past_k_time_interval, object_counts, k, device, dtype):
        cav_times = past_k_time_interval.to(device=device, dtype=dtype)
        cav_times = cav_times.view(-1, k)
        cav_times = torch.flip(cav_times, dims=[1])
        counts = object_counts.to(device=device).long()
        return torch.repeat_interleave(cav_times, counts, dim=0)

    def _build_motion_inputs(self, data_dict):
        past_boxes = data_dict['past_k_object_bbx']
        cur_boxes = data_dict['cur_object_bbx_debug']
        device = past_boxes.device
        dtype = next(self.parameters()).dtype
        past_boxes = past_boxes.to(device=device, dtype=dtype)
        cur_boxes = cur_boxes.to(device=device, dtype=dtype)
        k = past_boxes.shape[1]
        if k < 3:
            raise ValueError('CoBEVFlow motion prediction requires at least 3 history frames.')

        past_xy = past_boxes[:, :, :2]
        obj_coords = torch.flip(past_xy, dims=[1])
        obj_coords_norm = obj_coords - obj_coords[:, -1:, :]

        past_times = self._expand_cav_times(
            data_dict['past_k_time_interval'],
            data_dict['past_k_object_cav_num'],
            k,
            device,
            dtype)
        past_times_norm = past_times - past_times[:, -1:]

        speed = torch.zeros_like(obj_coords_norm)
        denom = past_times[:, 1:] - past_times[:, :-1]
        valid = torch.abs(denom) > 1e-6
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        speed[:, 1:, :] = (
            obj_coords_norm[:, 1:, :] - obj_coords_norm[:, :-1, :]
        ) / safe_denom.unsqueeze(-1)
        speed[:, 1:, :] = speed[:, 1:, :] * valid.unsqueeze(-1)

        obj_input = torch.cat([obj_coords_norm, speed], dim=-1).unsqueeze(0)

        last_dt = past_times_norm[:, -1:] - past_times_norm[:, -2:-1]
        valid_last = torch.abs(last_dt) > 1e-6
        safe_last_dt = torch.where(valid_last, last_dt, torch.ones_like(last_dt))
        query = obj_coords_norm[:, -1:, :] + (
            obj_coords_norm[:, -1:, :] - obj_coords_norm[:, -2:-1, :]
        ) * (0 - past_times[:, -1:]).unsqueeze(-1) / safe_last_dt.unsqueeze(-1)
        query = torch.where(valid_last.unsqueeze(-1), query, torch.zeros_like(query))
        query = query.unsqueeze(0)

        future_time = (-past_times[:, -1:]).view(1, -1, 1)
        gt_offset = cur_boxes[:, :2] - past_xy[:, 0, :2]
        return obj_input, query, past_times_norm, future_time, gt_offset
