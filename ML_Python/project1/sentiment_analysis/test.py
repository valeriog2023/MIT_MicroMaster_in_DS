import os
import sys
import time
import traceback
import project1 as p1
import numpy as np

verbose = False

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_real(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if res != exp_res:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def equals(x, y):
    if type(y) == np.ndarray:
        return  np.allclose(x,y)
        #return (x == y).all()
    return x == y

def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == tuple:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected\n", exp_res, ", \ngot: \n", res)
        return True

def check_array(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == list:
        log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def check_get_order():
    ex_name = "Get order"
    if check_list(
            ex_name, p1.get_order,
            [0], 1):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    if check_list(
            ex_name, p1.get_order,
            [1, 0], 2):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    log(green("PASS"), ex_name, "")


def check_hinge_loss_single():
    ex_name = "Hinge loss single"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_single,
            exp_res, feature_vector, label, theta, theta_0):
        return
    log(green("PASS"), ex_name, "")
    #
    #
    feature_vector = np.array([0.88340609, 0.9918406,  0.32655913, 0.73720716,
                               0.31410744, 0.6750514, 0.94656334, 0.75574279, 0.85217611, 0.57847598])
    label = 1.0
    theta = np.array([0.05659911, 0.05041133, 0.15311163, 0.06782354, 0.1591812,  0.07406843,
                               0.05282267, 0.06616008, 0.05867332, 0.08643401])
    theta_0 = 0.5
    exp_res = 0
    if check_real(
            ex_name, p1.hinge_loss_single,
            exp_res, feature_vector, label, theta, theta_0):
        return
    log(green("PASS"), ex_name, "")
    #


def check_hinge_loss_full():
    ex_name = "Hinge loss full"

    feature_vector = np.array([[1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_perceptron_single_update():
    ex_name = "Perceptron single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    exp_res = (np.array([0, 3]), -0.5)
    if check_tuple(
            ex_name, p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1
    exp_res = (np.array([0, 3]), 0)
    if check_tuple(
            ex_name + " (boundary case)", p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_perceptron():
    ex_name = "Perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")


def check_average_perceptron():
    ex_name = "Average perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([-0.5, 1]), 1.5)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([-0.25, 1.5]), 1.75)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos_single_update():
    ex_name = "Pegasos single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.4)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 1])
    label, theta, theta_0 = 1, np.array([-1, 1]), 1
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.08]), 1.1)
    if check_tuple(
            ex_name +  " (boundary case)", p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -2
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.9)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


    feature_vector = np.array([ 0.0828868, -0.10646208, 0.38343851, -0.01120626, 0.17100637, -0.48162327, -0.39878723, 0.47473096, 0.0726876, 0.23776261])
    label = 1
    L = 0.9219988080185197
    eta = 0.06285088414062623
    theta = np.array([-0.19790459, 0.237979,-0.08821794, -0.3205563, -0.17074328, 0.48665944, 0.01888054, -0.2812287, -0.26312399,  0.4170255 ])
    theta_0 = 2.0411281766814042
    exp_res = ( np.array([-0.1864363, 0.2241885, -0.0831058, -0.3019806, -0.1608490, 0.4584583, 0.0177864, -0.2649319, -0.2478764, 0.3928595]), 2.0411282 )
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


    feature_vector = np.array([ 0.47409908,  0.39364394, -0.368247,    0.17532051, -0.02671259,  0.39627102,
                   0.43695843,  0.27308634, -0.25161927, -0.28775879])
    label = -1
    L = 0.18243207167148656
    eta = 0.23058595970431084
    theta = np.array([-0.21322926, -0.37256956, -0.01975689, -0.14113785, -0.04824861, -0.08888422,
                     -0.14102607, -0.096532, 0.21157628,  0.32441515])
    theta_0 = -0.4662720332863288
    exp_res = ( np.array([-0.3135801, -0.4476657, 0.0659868, -0.1756272, -0.0400594, -0.1765197, -0.2358501, -0.1554411, 0.2606959, 0.3771213]), -0.6968580 )
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos():
    ex_name = "Pegasos"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    L = 0.2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    feature_matrix = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 1])
    T = 1
    L = 1
    exp_res = (np.array([1-1/np.sqrt(2), 1-1/np.sqrt(2)]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    log(green("PASS"), ex_name, "")


def check_classify():
    ex_name = "Classify"

    feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([1, 1, 1])
    if check_array(
            ex_name, p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    feature_matrix = np.array([[-1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([-1])
    if check_array(
            ex_name + " (boundary case)", p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


    feature_matrix = np.array([[-2.40016526e-01, -4.45381861e-02,  1.71652667e-01, -1.65950431e-02,
                     -4.28771679e-01, -1.51241996e-01,  4.19368465e-01,  2.37046048e-01,
                     -3.14701467e-01,  2.86083564e-01, -1.03300677e-01,  5.10433703e-03,
                     -2.06962232e-01, -9.19486646e-02, -3.77334790e-01,  8.13623741e-02,
                     -4.21754176e-01,  4.04175357e-01, -3.39395898e-01,  1.27254939e-01,
                      8.64984518e-02,  3.78088253e-01, -4.52350992e-01,  3.38306542e-01,
                      1.26983134e-01,  1.94751425e-01,  7.20131114e-02,  4.63747841e-01,
                     -1.64195031e-01, -2.42482648e-01, -2.35111725e-01,  1.89320413e-02,
                      8.29933195e-02, -4.59114432e-01, -4.07035872e-01, -8.89054754e-02,
                      4.97314747e-01, -3.66675158e-01, -4.35744880e-01, -2.21640556e-01,
                     -4.95715285e-01, -5.15812841e-02, -6.25216994e-02, -2.28281916e-01,
                      4.49196542e-01,  1.20625568e-01, -1.82534707e-01, -1.15482157e-01,
                     -3.80709905e-01, -3.80701221e-01],
                    [-4.12884581e-01,  3.42506758e-01,  3.79273912e-01, -4.95498484e-01,
                     -5.91834827e-02, -4.20833695e-01, -6.14064260e-02,  3.09690946e-01,
                      9.37556802e-02, -2.80294448e-01,  4.79993285e-01, -2.07614790e-01,
                      1.39383683e-01,  4.51183393e-01,  3.76943446e-01,  2.94250468e-02,
                      4.45763505e-01,  2.92106207e-01,  4.91320693e-01, -3.43693503e-01,
                     -3.69006174e-01,  1.44872118e-01,  4.61363051e-01,  2.64441613e-01,
                     -4.03038157e-01,  1.06685594e-01, -1.84234098e-01, -1.37438965e-01,
                      2.05830802e-01,  3.02481246e-01, -1.20470043e-01,  2.85635672e-01,
                     -1.83943204e-01,  9.11862298e-02, -9.14040272e-02, -1.00719181e-01,
                     -3.87835960e-01,  1.18647139e-01, -5.54131751e-02,  1.45365287e-01,
                      3.15490726e-01, -9.02066625e-03,  5.90520675e-02, -5.13517381e-02,
                     -4.09567795e-01,  1.94863408e-01,  4.43192878e-01,  3.45087789e-01,
                     -4.73575203e-01, -3.53278562e-01],
                    [-9.68402018e-02,  3.26896636e-01, -2.84736890e-02,  1.02057578e-04,
                      1.75829063e-01, -4.75167380e-01, -9.20916646e-02, -2.75317615e-01,
                     -2.02908733e-01,  4.27039687e-01,  4.84254300e-02, -2.97962988e-01,
                     -3.78520114e-01, -5.96822858e-02,  4.50823293e-01,  4.57285444e-01,
                      3.63737801e-01, -3.51017709e-01,  4.82606397e-02, -4.49381288e-01,
                      1.29357266e-01, -2.16167760e-01, -2.55863473e-01, -2.89527899e-01,
                      2.99910407e-01, -4.93339118e-01, -4.65850606e-01, -4.95582949e-01,
                     -3.21982054e-01, -2.49138587e-01, -4.51275356e-01, -8.72576582e-02,
                     -3.57871523e-01, -2.10368381e-02, -1.17436866e-02, -1.90144471e-01,
                     -3.89519859e-01,  3.87547524e-02, -1.51362476e-01,  2.33370527e-01,
                      4.04352276e-01, -3.35533505e-01, -1.25792794e-01, -4.85350733e-01,
                      3.84852501e-02, -3.85519744e-01,  8.28171619e-02,  3.11655373e-01,
                     -3.26900483e-01, -4.43779673e-01],
                    [ 3.61674175e-01, -2.44167142e-01, -2.25483426e-01,  1.53957444e-01,
                      1.95789077e-01,  2.18360173e-01, -1.75206063e-01, -3.65924263e-01,
                     -3.22547618e-01, -3.84386461e-01, -3.02037163e-01,  3.31106908e-01,
                      8.60772415e-02,  1.36794495e-01, -2.11807608e-01, -4.27900213e-01,
                      4.99759842e-01,  4.08066192e-01, -3.89496801e-01, -3.87532739e-02,
                      1.16201807e-01, -1.21725027e-01, -3.39243780e-01, -3.55098141e-01,
                      1.29033656e-01, -1.57749406e-01, -2.05539985e-01, -4.92497683e-01,
                      1.26661930e-01, -2.66997061e-01, -6.11468933e-02, -3.94755450e-01,
                      5.81719075e-02, -2.41179820e-01,  1.26841073e-01,  2.08172151e-01,
                     -1.78319311e-01,  1.36058480e-01,  4.86869445e-01,  3.01057133e-01,
                      2.67933242e-01, -9.72413953e-04, -4.30074750e-01, -3.16234095e-01,
                      4.83471763e-01,  9.78593031e-02, -3.55156059e-01,  3.23437143e-01,
                     -1.02171936e-01,  3.28666244e-01],
                    [-5.89159777e-02,  1.40760118e-01,  3.08085936e-01,  3.52377155e-01,
                     -3.47169222e-01, -8.79153794e-02, -1.47349767e-01, -4.81656386e-01,
                      1.88730682e-01, -3.56465332e-01, -4.68963304e-01, -2.87563617e-01,
                      3.88251560e-01, -1.95062683e-01, -4.60338264e-01,  4.96994468e-01,
                      2.01895886e-01, -3.48111888e-01, -2.45653172e-01,  3.32840111e-01,
                      4.89474168e-01, -7.48349830e-02,  2.87295111e-01,  2.22675528e-01,
                      2.63463920e-01,  2.97626954e-01,  2.21332154e-01, -4.11505091e-01,
                      3.29706781e-01,  3.19931773e-01,  4.08736737e-02,  1.90233283e-01,
                      4.45969878e-01, -1.71261849e-01,  1.13712749e-01,  2.03479931e-01,
                      3.13668872e-01, -1.41175083e-01,  3.02453784e-01, -3.66476377e-02,
                      4.76126294e-02, -4.85124889e-01,  4.06949424e-01, -2.38587340e-01,
                      8.83523193e-02,  3.17070835e-01,  2.01924315e-01, -4.00217473e-01,
                      2.69465692e-01, -3.96340501e-01]])
    theta = np.array([0.54832355, 0.52233932,  0.56545898, 0.45944794, 0.0362408 , 0.06416029,
                      0.20846829, 0.78798376,  0.7803661 , 0.63420283, 0.18866799, 0.27889163,
                      0.26032736, 0.0458161 ,  0.96557009, 0.11552231, 0.09532557, 0.31622758,
                      0.29625311, 0.84852633,  0.00428204, 0.29187582, 0.35963862, 0.28579249,
                      0.22172312, 0.25421042,  0.78439406, 0.27836737, 0.51533251, 0.76167303,
                      0.89164211, 0.66663657,  0.40128352, 0.02145718, 0.6271197 , 0.21471251,
                      0.37941618, 0.1736163 ,  0.35802917, 0.05514026, 0.03428902, 0.49037774,
                      0.26633233, 0.29110444,  0.45101488, 0.68797688, 0.99287213, 0.30115188,
                      0.00168071, 0.66877973])
    theta_0 = 0.9811939767742291
    exp_res = np.array([ 1.,  1., -1., -1. , 1.])
    if check_array(
            ex_name + " (boundary case)", p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return
    ex_name = "Classify 2"
    log(green("PASS"), ex_name, "")





    feature_matrix = np.array([[-0.18035266,  0.26182282, -0.02490685, -0.24513227,  0.08676946,  0.4402173,
                     -0.13968746, -0.28835115,  0.1422221 ,  0.31607662,  0.28156075,  0.40832034,
                     -0.37398398, -0.33587911, -0.33889814,  0.48595429, -0.38283135, -0.37234996,
                      0.32518948, -0.08055446,  0.11320608,  0.30988206,  0.44730623,  0.36312439,
                      0.24493113,  0.15951095, -0.26253184, -0.27117708,  0.21571212, -0.39067283,
                     -0.23965965, -0.46529878,  0.45516784, -0.3474515 , -0.41630842, -0.38516831,
                     -0.42956737,  0.35085962,  0.32909403,  0.16432321,  0.11027176,  0.00103193,
                      0.08252023,  0.45499068, -0.34899748, -0.10111043,  0.48871237, -0.17093076,
                      0.18678596,  0.18324789],
                    [ 0.44996796,  0.23229094,  0.13440199,  0.13924549, -0.41605203, -0.34574386,
                      0.13465765, -0.49996126, -0.09336038,  0.01351293,  0.1520306 ,  0.0524442,
                      0.31162   , -0.15702619, -0.00383153, -0.1656651 ,  0.19556177, -0.23283112,
                     -0.41200543, -0.30969183, -0.41091482,  0.13401141, -0.36574868,  0.21672832,
                      0.10277825, -0.37286166, -0.3916026 ,  0.12363316,  0.05964953, -0.16540919,
                      0.25828949,  0.10218957,  0.07035183,  0.34895141,  0.450754  , -0.14857216,
                      0.18692656,  0.24661765,  0.33222289, -0.26432006,  0.17595992, -0.44605399,
                      0.41979449, -0.20107196, -0.23100041, -0.49018901,  0.46762962,  0.23213349,
                      0.31423341, -0.38742342],
                    [ 0.15013991, -0.47759512,  0.45363303, -0.1526139 , -0.26043497, -0.06088544,
                      0.14853137,  0.44002926, -0.40115314,  0.15723225,  0.05901693,  0.4837523,
                      0.07934339, -0.23427349,  0.47458004,  0.26968787,  0.12364505,  0.465662,
                     -0.40734289,  0.42792691,  0.31656315, -0.34143036, -0.18590585, -0.46133916,
                      0.41594787, -0.04923342,  0.37025776, -0.41205106,  0.41290009, -0.32282482,
                      0.15313751, -0.46742896, -0.27983179,  0.09304276, -0.4291053 ,  0.0623168,
                      0.27591568,  0.09816399, -0.4078024 ,  0.47623273,  0.0215819 ,  0.27063166,
                     -0.12444878,  0.28260355, -0.20586239, -0.11930517,  0.46577571, -0.44563444,
                      0.40619422, -0.0478033 ],
                    [ 0.07728704, -0.09031609,  0.0532445 , -0.26400033, -0.31351206,  0.00250126,
                      0.21832355, -0.43631591, -0.33281485, -0.26693043, -0.0244386 , -0.39166138,
                      0.04136556, -0.48009866, -0.21232866,  0.41503207, -0.39094728,  0.01887381,
                     -0.26048627, -0.22624567,  0.08003307, -0.15361324,  0.31481394,  0.0748178,
                     -0.1571395 ,  0.12399195, -0.41764018,  0.32435427,  0.32285913,  0.3399522,
                      0.06149951, -0.16463971,  0.40858344, -0.38258137,  0.29323953, -0.35693253,
                      0.29859731, -0.03682619,  0.10731398, -0.15471703,  0.23849716, -0.01025152,
                      0.10054125, -0.10913378,  0.15242878,  0.18374257, -0.18166854,  0.12902091,
                     -0.48330393,  0.01728538],
                    [ 0.42821606,  0.37887446,  0.22401061,  0.34499184, -0.0615985 ,  0.44184543,
                      0.18246156, -0.19547608, -0.33921677,  0.48872889,  0.3149153 ,  0.41560942,
                      0.27345992,  0.02149755, -0.11880052, -0.14552205,  0.09915317,  0.26437266,
                     -0.37727538,  0.21478246, -0.43453274, -0.1947181 , -0.14554413, -0.3489313,
                     -0.40561929, -0.23845455,  0.35591581, -0.40869351, -0.17365439,  0.15603768,
                     -0.10943747, -0.11893032, -0.19274068, -0.37286423, -0.21194229,  0.18063736,
                      0.1260944 ,  0.00928053, -0.06223605,  0.16161273,  0.16470672, -0.02195013,
                     -0.22547272,  0.00879609,  0.49132265,  0.36184057, -0.03281353,  0.12347262,
                     -0.26969705, -0.29814545]])
    theta = np.array([0.21679242, 0.76494502, 0.25297135, 0.61627693, 0.21557389, 0.16486937,
                      0.49075618, 0.40864023, 0.10799801, 0.36850288, 0.15843785, 0.18157201,
                      0.71112424, 0.82285038, 0.33619448, 0.0666579 , 0.76642279, 0.12528888,
                      0.44930238, 0.74813141, 0.30538793, 0.55951553, 0.6923939 , 0.88780238,
                      0.0291748 , 0.4970397 , 0.39234796, 0.94530304, 0.81690804, 0.41583167,
                      0.15332822, 0.21451381, 0.6205286 , 0.50690868, 0.40033757, 0.72695989,
                      0.8372106 , 0.34997579, 0.07552982, 0.70549423, 0.0718128 , 0.09405372,
                      0.90983285, 0.62462861, 0.51279751, 0.08008669, 0.19517514, 0.69350241,
                      0.25877189, 0.02354368])
    theta_0 = -0.001
    exp_res = np.array([-1.,  1., -1., -1.,  1.])
    if check_array(
            ex_name + " (boundary case)", p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return
    ex_name = "Classify 3"
    log(green("PASS"), ex_name, "")







def check_classifier_accuracy():
    ex_name = "Classifier accuracy"

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.perceptron,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T):
        return

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    L=0.2
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.pegasos,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T, L=L):
        return

    log(green("PASS"), ex_name, "")







































def check_bag_of_words():
    ex_name = "Bag of words"

    texts = [
        "He loves to walk on the beach",
        "There is nothing better"]

    try:
        res = p1.bag_of_words(texts)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return
    if not type(res) == dict:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return

    vals = sorted(res.values())
    exp_vals = list(range(len(res.keys())))
    if not vals == exp_vals:
        log(red("FAIL"), ex_name, ": wrong set of indices. Expected: ", exp_vals, " got ", vals)
        return

    log(green("PASS"), ex_name, "")

    keys = sorted(res.keys())
    exp_keys = ['beach', 'better', 'he', 'is', 'loves', 'nothing', 'on', 'the', 'there', 'to', 'walk']
    stop_keys = ['beach', 'better', 'loves', 'nothing', 'walk']

    if keys == exp_keys:
        log(yellow("WARN"), ex_name, ": does not remove stopwords:", [k for k in keys if k not in stop_keys])
    elif keys == stop_keys:
        log(green("PASS"), ex_name, " stopwords removed")
    else:
        log(red("FAIL"), ex_name, ": keys are missing:", [k for k in stop_keys if k not in keys], " or are not unexpected:", [k for k in keys if k not in stop_keys])


def check_extract_bow_feature_vectors():
    ex_name = "Extract bow feature vectors"
    texts = [
        "He loves her ",
        "He really really loves her"]
    keys = ["he", "loves", "her", "really"]
    dictionary = {k:i for i, k in enumerate(keys)}
    exp_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 1]])
    non_bin_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 2]])


    try:
        res = p1.extract_bow_feature_vectors(texts, dictionary)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return

    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return

    log(green("PASS"), ex_name)

    if (res == exp_res).all():
        log(yellow("WARN"), ex_name, ": uses binary indicators as features")
    elif (res == non_bin_res).all():
        log(green("PASS"), ex_name, ": correct non binary features")
    else:
        log(red("FAIL"), ex_name, ": unexpected feature matrix")
        return

def main():
    log(green("PASS"), "Import project1")
    try:
        check_get_order()
        check_hinge_loss_single()
        check_hinge_loss_full()
        check_perceptron_single_update()
        check_perceptron()
        check_average_perceptron()
        check_pegasos_single_update()
        check_pegasos()
        check_classify()
        check_classifier_accuracy()
        check_bag_of_words()
        check_extract_bow_feature_vectors()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()
