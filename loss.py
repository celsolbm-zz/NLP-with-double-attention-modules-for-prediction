class QueueLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2):
        super(QueueLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, outputs, targets,att1,att2,att3,gamma, mask=None):
        target_mask = (torch.arange(2).unsqueeze(0).expand(targets.size(0), -1).to(device) == targets.unsqueeze(1)).float()
        lambdas = outputs.exp()
        prob_expo = 1-(-lambdas/2).exp()
        loss =  torch.cat([(-(1.0 - prob_expo+0.001).log())*(1+prob_expo)**gamma, \
                           (-(prob_expo+0.001).log())*(2-prob_expo)**gamma], -1) * target_mask
        att1T = att1.transpose(1,2)
        att2T = att2.transpose(1,2)
        att3T = att3.transpose(1,2)
        loss = loss.sum()
        penal = 1
        loss = loss + penal* (torch.norm(att1@att1T - torch.eye(att1.size(1)).cuda())/att1.size(0) \
             + torch.norm(att2@att2T - torch.eye(att2.size(1)).cuda())/att2.size(0) \
             + torch.norm(att3@att3T - torch.eye(att3.size(1)).cuda())/att3.size(0))
        return loss