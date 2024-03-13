import torch
import time
from tqdm import tqdm

from torch.autograd import Variable
from utils.metrics import AverageMeter
from utils.loss.utils import _sigmoid

def train_model(opts, model, train_dataloder, optimizer, criternions, epoch,gen_pcs_tem,gen_pcs_sea):
    # total loss
    losses_total = AverageMeter()
    # regression loss
    losses_reg_hm = AverageMeter()
    losses_reg_loc = AverageMeter()
    losses_reg_z = AverageMeter()

    # train model
    model.train()
    
    with tqdm(enumerate(train_dataloder), total=len(train_dataloder), ncols=opts.ncols) as t:
        for i, data in t:
            # 1. get inputs
            # data : {
            #     'template_pc':      templates_PC,
            #     'search_pc':        target_PC,
            #     'heat_map':         hot_map,
            #     'index_center':     index_center,
            #     'z_axis':           z_axis,
            #     'index_offsets':    index_offsets,
            #     'local_offsets':    local_offsets,
            # }
            torch.cuda.synchronize()
            # data['template_pc'] =gen_pcs_tem[i]
            # data['search_pc'] = gen_pcs_sea[i]
            # data_template_pc=torch.cat((data['template_pc'],gen_pcs_tem[i]),dim=2)
            # data_search_pc = torch.cat((data['search_pc'], gen_pcs_sea[i]), dim=2)
            # x1_values1 =data_template_pc[:, :, 0]
            # x2_values1 = data_template_pc[:, :, 3]
            # y1_values1 = data_template_pc[:, :, 1]
            # y2_values1 = data_template_pc[:, :, 4]
            # z1_values1 = data_template_pc[:, :, 2]
            # z2_values1 = data_template_pc[:, :, 5]
            #
            # x1_values2 = data_search_pc[:, :, 0]
            # x2_values2 = data_search_pc[:, :, 3]
            # y1_values2 = data_search_pc[:, :, 1]
            # y2_values2 = data_search_pc[:, :, 4]
            # z1_values2 = data_search_pc[:, :, 2]
            # z2_values2 = data_search_pc[:, :, 5]
            #
            # x_mean1 = (x1_values1 + x2_values1) / 2
            # y_mean1 = (y1_values1 + y2_values1) / 2
            # z_mean1 = (z1_values1 + z2_values1) / 2
            #
            # x_mean2 = (x1_values2 + x2_values2) / 2
            # y_mean2 = (y1_values2 + y2_values2) / 2
            # z_mean2 = (z1_values2 + z2_values2) / 2
            #
            # data['template_pc'][:, :, 0] = x_mean1
            # data['template_pc'][:, :, 1] = y_mean1
            # data['template_pc'][:, :, 2] = z_mean1
            #
            # data['search_pc'][:, :, 0] = x_mean2
            # data['search_pc'][:, :, 1] = y_mean2
            # data['search_pc'][:, :, 2] = z_mean2

            # data[]
            data = {key: Variable(value, requires_grad=False).to(opts.device) for key, value in data.items()}
            
            pred_hm, pred_loc, pred_z_axis,pred_confidence = model(data['template_pc'], data['search_pc'],gen_pcs_tem[i])
            #pred_hm, pred_loc, pred_z_axis,pred_confidence = model(data['template_pc']+gen_pcs_tem[i], data['search_pc'])

            pred_hm = _sigmoid(pred_hm)

            # print("score:",pred_confidence)

            #change pred_nm to (36)

            # 3. calculate loss
            loss_reg_hm = criternions['hm'](pred_hm, data['heat_map'])
            loss_reg_loc = criternions['loc'](pred_loc, data['index_offsets'], data['local_offsets'])
            loss_reg_z = criternions['z_axis'](pred_z_axis, data['index_center'], data['z_axis'])
            # total loss
            total_loss = 1.0*loss_reg_hm + 1.0*loss_reg_loc + 2.0*loss_reg_z

            # 4. calculate gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()

            # 5. update infomation
            # 5.1 update training error
            # regression loss
            losses_reg_hm.update(1.0*loss_reg_hm.item())
            losses_reg_loc.update(1.0*loss_reg_loc.item())
            losses_reg_z.update(2.0*loss_reg_z.item())
            # total loss
            losses_total.update(total_loss.item())

            lr = optimizer.param_groups[0]['lr']
            t.set_description(f'Train {epoch}: '
                            f'Loss:{losses_total.avg:.3f} '
                            f'Regress:({losses_reg_hm.avg:.4f}, '
                            f'{losses_reg_loc.avg:.4f}, '
                            f'{losses_reg_z.avg:.3f}), '
                            f'lr:{1000*lr:.3f} '
                            )

    return losses_total.avg

def valid_model(opts, model, valid_dataloder, criternions, epoch,gen_pcs_tem,gen_pcs_sea):
    # total loss
    losses_total = AverageMeter()
    # regression loss
    losses_reg_hm = AverageMeter()
    losses_reg_loc = AverageMeter()
    losses_reg_z = AverageMeter()

    # evaluate model
    model.eval()

    with tqdm(enumerate(valid_dataloder), total=len(valid_dataloder), ncols=opts.ncols) as t:
        with torch.no_grad():
            end = time.time()
            for i, data in t:
                # 1. get inputs
                data = {key: Variable(value, requires_grad=False).to(opts.device) for key, value in data.items()}
            
                # 2. calculate outputs
                pred_hm, pred_loc, pred_z_axis = model(data['template_pc'], data['search_pc'],gen_pcs_tem[i],gen_pcs_sea[i])
                pred_hm = _sigmoid(pred_hm)


                # 3. calculate loss
                loss_reg_hm = criternions['hm'](pred_hm, data['heat_map'])
                loss_reg_loc = criternions['loc'](pred_loc, data['index_offsets'], data['local_offsets'])
                loss_reg_z = criternions['z_axis'](pred_z_axis, data['index_center'], data['z_axis'])
                # total loss
                total_loss = 1.0*loss_reg_hm + 1.0*loss_reg_loc + 2.0*loss_reg_z

                # 4. update infomation
                # 4.1 update training error
                # regression loss
                losses_reg_hm.update(1.0*loss_reg_hm.item())
                losses_reg_loc.update(1.0*loss_reg_loc.item())
                losses_reg_z.update(2.0*loss_reg_z.item())
                # total loss
                losses_total.update(total_loss.item())

                t.set_description(  f'Test  {epoch}: '
                                    f'Loss:{losses_total.avg:.3f} '
                                    f'Regress:({losses_reg_hm.avg:.4f}, '
                                    f'{losses_reg_loc.avg:.4f}, '
                                    f'{losses_reg_z.avg:.3f}), '
                                    )

    return losses_total.avg