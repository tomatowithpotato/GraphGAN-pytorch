import src.GraphGAN.config as config
from src.evaluation import link_prediction as lp
from src.evaluation import node_classification as nc
from src.evaluation import recommendation as rcmd

def lp_train_info(item, embed_filename, test_filename, test_neg_filename, n_node, n_embed, rcmd=None):
    lpe = lp.LinkPredictEval(embed_filename, test_filename, test_neg_filename, n_node, n_embed)

    if rcmd is None:
        result = lpe.eval_link_prediction(emd=item.embedding_matrix.detach().numpy())
    else:
        result = lpe.eval_rcmd_link_prediction(rcmd=rcmd, emd=item.embedding_matrix.detach().numpy())

    val_acc, val_macro = result["acc"], result["macro"]
    val_info = "val_acc: {} val_macro: {}".format(val_acc, val_macro)
    return val_info

def eval_info(eval_result, i):
    head_info = config.app + "\t" +config.modes[i]
    acc_info = "acc: {}\t".format(eval_result["acc"])
    macro_info = "macro: {}\t".format(eval_result["macro"])
    return head_info, acc_info, macro_info

def eval_info_rcmd(eval_result, K, i):
    head_info = config.app + "\t" + config.modes[i] + "\trcmd_K: {}".format(K)
    precision_info = "precision: {}\t".format(eval_result["precision"])
    recall_info = "recall: {}\t".format(eval_result["recall"])
    return head_info, precision_info, recall_info


def eval_while_train(gGAN, item_name, epoch, save=False):
    lr=(config.lr_gen, config.lr_dis)
    i, item = (0, gGAN.generator) if item_name == 'gen' else (1, gGAN.discriminator)
    val_info = ""

    if config.app == "link_prediction":           
        lp_info = lp_train_info(item, config.emb_filenames[i], config.test_filename, config.test_neg_filename,
                                gGAN.n_node, config.n_emb)
        val_info = lp_info
    elif config.app == "node_classification":
        # lpe = lp.LinkPredictEval(
        #         config.emb_filenames[i], config.train_filename, config.neg_filename, gGAN.n_node, config.n_emb)
        # lp_result = lpe.eval_link_prediction()
        pass
    elif config.app == "recommendation":
        lp_info = lp_train_info(item, config.emb_filenames[i], config.rcmd_test_filename, config.rcmd_test_neg_filename, 
                                gGAN.n_node, config.n_emb, rcmd=gGAN.rcmd)
        val_info = lp_info
    
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
        results.append("\n" + "-"*50 + "\n")
        for i in range(2):
            with open(config.train_detail_filename[i], mode="a+") as f:
                f.writelines("\n" + "-"*50 + "\n")
    
    results.append("epoch: " + str(epoch) + "\n")

    for i in range(len(config.modes)):
        if config.app == "link_prediction":
            lpe = lp.LinkPredictEval(config.emb_filenames[i], 
                                     config.test_filename, 
                                     config.test_neg_filename, 
                                     gGAN.n_node, 
                                     config.n_emb)
            lp_result = lpe.eval_link_prediction()

            head_info, acc_info, macro_info = eval_info(lp_result, i)

            results.append(head_info + "\t" + acc_info + macro_info + "lr: " + str(lr[i]) + "\n")

        elif config.app == "node_classification":
            #注意，对于节点分类不分训练集和测试集，train即是全集
            lpe = lp.LinkPredictEval(config.emb_filenames[i], 
                                     config.train_filename, 
                                     config.neg_filename, 
                                     gGAN.n_node, 
                                     config.n_emb)
            lp_result = lpe.eval_link_prediction()

            nce = nc.NodeClassificationEval(config.emb_filenames[i], 
                                            gGAN.labels_matrix, 
                                            gGAN.n_node, 
                                            config.n_emb, 
                                            gGAN.n_classes)
            nc_result = nce.eval_node_classification()

            head_info, acc_info, macro_info = eval_info(nc_result, i)
            
            lp_acc_info = "lp_acc: {}\t".format(lp_result["acc"])

            results.append(head_info + "\t" + acc_info + macro_info + lp_acc_info + "lr: " + str(lr[i]) + "\n")

        elif config.app == "recommendation":
            lpe = lp.LinkPredictEval(config.emb_filenames[i], 
                                     config.rcmd_test_filename, 
                                     config.rcmd_test_neg_filename, 
                                     gGAN.n_node, 
                                     config.n_emb)
            lp_result = lpe.eval_rcmd_link_prediction(rcmd=gGAN.rcmd)

            rcmde = rcmd.Recommendation(config.emb_filenames[i], 
                                        gGAN.n_node, 
                                        config.n_emb,
                                        rcmd=gGAN.rcmd)
            
            lp_acc_info = "lp_acc: {}\t".format(lp_result["acc"])
            
            for K in config.rcmd_K:
                rcmd_result = rcmde.eval_rcmd_K_movie(K)
                head_info, precision_info, recall_info = eval_info_rcmd(rcmd_result, K, i)
                results.append(head_info + "\t" + precision_info + recall_info + lp_acc_info + "lr: " + str(lr[i]) + "\n")


    with open(config.result_filename, mode="a+") as f:
        f.writelines(results)