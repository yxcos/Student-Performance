import torch
import torch.optim as optim
from model import *
import utils
from thop import profile 


class trainer():
    def __init__(self, args, scaler, adj, history, num_of_vertices,
                 in_dim, hidden_dims, first_layer_embedding_size, out_layer_dim,
                 log, lrate, device, activation='GLU', use_mask=True, max_grad_norm=5,
                 lr_decay=False, temporal_emb=True, spatial_emb=True, horizon=12, strides=3):
  
        super(trainer, self).__init__()

        self.model = USTGA(
            adj=adj,
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            first_layer_embedding_size=first_layer_embedding_size,
            out_layer_dim=out_layer_dim,
            activation=activation,
            use_mask=use_mask,
            temporal_emb=temporal_emb,
            spatial_emb=spatial_emb,
            horizon=horizon,
            strides=strides
        )

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.model_parameters_init()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, eps=1.0e-8, weight_decay=0, amsgrad=False)

        if lr_decay:
            utils.log_string(log, 'Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.scaler = scaler
        self.clip = max_grad_norm

        utils.log_string(log, "Parameters: {:,}".format(utils.count_parameters(self.model)))
        utils.log_string(log, 'GPU Occupancy:{:,}'.format(torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))

    def model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, input, real_val):
      
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)  

        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mae = utils.masked_mae(predict, real_val).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()

        return loss.item(), mae, mape, rmse

    def evel(self, input, real_val):
      
        self.model.eval()

        output = self.model(input)  

        predict = self.scaler.inverse_transform(output)  

        mae = utils.masked_mae(predict, real_val, 0.0).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()

        return mae, mape, rmse