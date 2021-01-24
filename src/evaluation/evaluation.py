import src.GraphGAN.config as config
from src.evaluation import link_prediction as lp
from src.evaluation import node_classification as nc
from src.evaluation import recommendation as rcmd

def eval_while_train(gGAN, item_name, epoch, save=False):
    lr=(config.lr_gen, config.lr_dis)
    i, item = (0, gGAN.generator) if item_name == 'gen' else (1, gGAN.discriminator)
    val_info = ""

    if config.app == "link_prediction":           
        lpe = lp.LinkPredictEval(
                config.emb_filenames[i], config.test_filename, config.test_neg_filename, 
                gGAN.n_node, config.n_emb)
        result = lpe.eval_link_prediction(emd=item.embedding_matrix.detach().numpy())
        val_acc, val_macro = result["acc"], result["macro"]
        val_info = "val_acc: {} val_macro: {}".format(val_acc, val_macro)
    elif config.app == "node_classification":
        # lpe = lp.LinkPredictEval(
        #         config.emb_filenames[i], config.train_filename, config.neg_filename, gGAN.n_node, config.n_emb)
        # lp_result = lpe.eval_link_prediction()
        pass
    elif config.app == "recommendation":
        rcmd_lp_e = rcmd.Recommendation(
                config.emb_filenames[i], config.test_filename, config.test_neg_filename, 
                gGAN.n_node, config.n_emb, user_max=gGAN.user_max)
        result = rcmd_lp_e.eval_rcmd_lp(emd=item.embedding_matrix.detach().numpy())
        val_acc, val_macro = result["acc"], result["macro"]
        val_info = "val_acc: {} val_macro: {}".format(val_acc, val_macro)
    
    if save == True and val_info != '':
        split = ' ' * 3
        lr_info = "lr: {}".format(str(lr))
        head_info = ' '.join([config.app, config.modes[i]])
        result = split.join([head_info, val_info, lr_info, '\n'])
        with open(config.train_detail_filename[i], mode="a+") as f:
            f.writelines(result)
    return val_info

def eval(gGAN, epoch):
    lr=(config.lr_gen, config.lr_dis)
    results = []

    if epoch == "pre_train":
        results.append("\n")
        for i in range(2):
            with open(config.train_detail_filename[i], mode="a+") as f:
                f.writelines("\n")
    
    results.append("epoch: " + str(epoch) + "\n")
    if config.app == "link_prediction":
        for i in range(2):
            lpe = lp.LinkPredictEval(
                config.emb_filenames[i], config.test_filename, config.test_neg_filename, gGAN.n_node, config.n_emb)
            result = lpe.eval_link_prediction()

            head_info = config.app + "\t" +config.modes[i]
            acc_info = "acc: {}\t".format(result["acc"])
            macro_info = "macro: {}\t".format(result["macro"])
            results.append(head_info + "\t" + acc_info + macro_info + "lr: " + str(lr[i]) + "\n")
    elif config.app == "node_classification":
        for i in range(2):
            lpe = lp.LinkPredictEval(
                config.emb_filenames[i], config.train_filename, config.neg_filename, gGAN.n_node, config.n_emb)
            lp_result = lpe.eval_link_prediction()

            nce = nc.NodeClassificationEval(
                config.emb_filenames[i], gGAN.labels_matrix, gGAN.n_node, config.n_emb, gGAN.n_classes)
            nc_result = nce.eval_node_classification()

            head_info = config.app + "\t" +config.modes[i]
            acc_info = "acc: {}\t".format(nc_result["acc"])
            macro_info = "macro: {}\t".format(nc_result["macro"])
            lp_acc_info = "lp_acc: {}\t".format(lp_result["acc"])
            results.append(head_info + "\t" + acc_info + macro_info + lp_acc_info + "lr: " + str(lr[i]) + "\n")

    with open(config.result_filename, mode="a+") as f:
        f.writelines(results)