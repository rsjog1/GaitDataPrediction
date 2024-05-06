import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# This function loads the signal measusrementa and labels, and splits it into time and values.
def loadTrial(dataFolder,id):
    x = np.genfromtxt('{}Trial{:02d}_x.csv'.format(dataFolder,id),delimiter=',')
    xt = x[:,0]
    xv = x[:,1:]
    y = np.genfromtxt('{}Trial{:02d}_y.csv'.format(dataFolder,id),delimiter=',')
    yt = y[:,0]
    yv = y[:,1].astype(int)

    # Returning x measurements and y labels
    return xt, xv, yt, yv

# This function extracts features from the measurements.
def extractFeat(xt,xv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    featList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting features
        xWin = xv[(xt>=t0)*(xt<=t1),:]
        f1 = np.mean(xWin,axis=0)
        f2 = np.std(xWin,axis=0)

        # Storing the features
        featList.append(np.concatenate((f1,f2)))

        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    featList = np.array(featList)

    return tList, featList

# This function returns the mode over a window of data to make it compatible with the features
# extracted.
def extractLabel(yt,yv,winSz,timeStart,timeEnd,timeStep):
    tList = []
    yList = []

    # Specifying the initial window for extracting features
    t0 = timeStart
    t1 = t0+winSz

    while(t1<=timeEnd):
        # Using the middle time of the window as a reference time
        tList.append((t0+t1)/2)

        # Extracting features
        yWin = yv[(yt>=t0)*(yt<=t1)]
        
        # Storing the features
        yList.append(st.mode(yWin).mode)

        # Updating the window by shifting it by the step size
        t0 = t0+timeStep
        t1 = t0+winSz

    tList = np.array(tList)
    yList = np.array(yList)

    return tList, yList



def summary(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)



def summaryPerf(y, yHat, yTrain, yTrainHat):
    cm = metrics.confusion_matrix(y, yHat, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walk Hard','Down Stairs','Up Stairs','Walk Soft'])
    disp.plot()

    print('Training:  Acc = {:4.3f}'.format(metrics.accuracy_score(yTrain,yTrainHat)))
    print('Non-Train: Acc = {:4.3f}'.format(metrics.accuracy_score(y,yHat)))
    print('Training:  BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(yTrain,yTrainHat)))
    print('Non-Train: BalAcc = {:4.3f}'.format(metrics.balanced_accuracy_score(y,yHat)))

