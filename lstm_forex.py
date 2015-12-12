'''
LSTM RNN for forex predictions
Based on sentiment analysis lstm found in deeplearning tutorials
'''
from collections import OrderedDict
import copy
import cPickle as pkl
import random
import sys
import time
import pdb
import os
import logging

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse
from theano import config


from forex import read_data, prepare_data

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

#### rectified linear unit
def ReLU(x):
    y = tensor.maximum(0.0, x)
    return(y)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_input'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype('float32')
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype('float32')
    params['b'] = numpy.zeros((options['ydim'],)).astype('float32')

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W.astype('float32')
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U.astype('float32')
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype('float32')

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    #assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        #c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        #TODO: I think this don't apply since is made to avoid sequences smaller tan max_len
        #h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(0., n_samples,
                                                           dim_proj),
                                              tensor.alloc(0., n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}

def mom_sgd(lr, tparams, grads, x, y, cost):
    """ Momentum Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """

    updates = OrderedDict()

    mom = tensor.scalar(name='mom')
    gmomshared = [theano.shared(p.get_value(), name='%s_mom_grad' %k)
        for k,p in tparams.iteritems()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    for gm,gp in zip(gmomshared,gshared):
        updates[gm] = mom*gm - (1.0 - mom) * lr * gp
    #gmomup = [(gm, mom*gm - (1.0 - mom) * lr * gp) for gm,gp in
    #    zip(gmomshared, gshared)]
    
    #pup = [(p, p + gm) for p, gm in zip(tparams.values(), gmomup)]
    for p,gm in zip(tparams.values(), gmomshared):
        updates[p] = p + updates[gm]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr,mom], [], updates=updates,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup+rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update',
                               mode='DebugMode')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(1234)

    # Used for dropout.
    use_noise = theano.shared(numpy.float32(0.))

    x = tensor.tensor3('x', dtype='float32')
    #mask = tensor.matrix('mask', dtype='float32')
    y = tensor.vector('y', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    n_dim = x.shape[2]

    emb = tensor.dot(x,tparams['Wemb'])
    #emb = tensor.nnet.sigmoid(emb)
    #emb = ReLU(emb)

    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng)

    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder']
                                            )
    

    if options['encoder'] == 'lstm' and options['sum_pool'] == True:
        proj = proj.sum(axis=0)
        proj = proj / options['n_iter'] 
    else:
        proj = proj[-1]
    #if options['use_dropout']:
    #    proj = dropout_layer(proj, use_noise, trng)

    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U'])+tparams['b'])
    #pred = tensor.nnet.sigmoid(tensor.dot(proj, tparams['U'])\
    #        + tparams['b'])
    pred = tensor.dot(proj, tparams['U']) + tparams['b']

    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    #f_pred = theano.function(x, pred.argmax(axis=1), name='f_pred')

    cost = tensor.mean((y-pred.T)**2)

    #cost = -tensor.log(pred[tensor.arange(n_samples), y] + 1e-8).mean()

    return use_noise, x, y, f_pred_prob, cost


def pred_probs(f_pred, prepare_data, data, model_options, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data)

    x,y = prepare_data(data, numpy.array([]), 
                                model_options['n_iter'], model_options['n_input'])
    pred = f_pred(x) 

    return pred

def pred_error(f_pred, prepare_data, data, iterator, model_options, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        # TODO: This is not very efficient I should check
        x,  y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  model_options['n_iter'],model_options['n_input'])


        preds = f_pred(x)
        targets = numpy.array(data[1])[valid_index]
        # or tensor.sum
        valid_err += ((targets-preds.T)**2).sum()
    #valid_err = 1. - numpy.float32(valid_err) / len(data[0])
    valid_err = numpy_floatX(valid_err) / len(data[0])    

    return valid_err


def R_score(f_pred, prepare_data, data, iterator, model_options, verbose=False):
    """
    Compute R score
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    denom = 0
    data_mean = numpy.array(data[1]).mean()
    for _, valid_index in iterator:
        # TODO: This is not very efficient I should check
        x,  y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  model_options['n_iter'],model_options['n_input'])


        preds = f_pred(x)
        targets = numpy.array(data[1])[valid_index]
        valid_err += tensor.sum((targets-preds.T)**2)
        denom += ((numpy.array(data[1]) - data_mean)**2).sum()
    #valid_err = 1. - numpy.float32(valid_err) / len(data[0])
    valid_err = 1. - (valid_err / denom) 

    return valid_err.eval()


def backforecast(f_pred, data, model_options):
    """
    Compute the amount of times in which
    the RNN correctly predict a up or 
    down trend
    """
    # TODO: Use the prepare data
    x, y = prepare_data(data[0], data[1], model_options['n_iter'],
        model_options['n_input'])

    targets =  (y > x[-1,:,0])
    preds = f_pred(numpy.asarray(x,dtype='float32'))
    preds_up = (preds[:,0] > x[-1,:,0])
    err = (targets <> preds_up).sum()

    ret = float(err) / float(len(data[0]))
    return ret

def predict_lstm(input, model_options):
    
    params = init_params(model_options)

    load_params(model_options['saveto'], params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x,
     y, f_pred_prob, cost) = build_model(tparams, model_options)

    preds = pred_probs(f_pred_prob, prepare_data, input, model_options)

    return preds


def train_lstm(
    #dim_proj=32,  # word embeding dimension and LSTM number of hidden units.
    dim_proj=124,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=150,  # The maximum number of epoch to run
    dispFreq=40,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.1,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_input = 4,  # Vocabulary size
    optimizer=mom_sgd,  # sgd,mom_sgs, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    tick='hour',
    validFreq=5,  # Compute the validation error after this number of update.
    saveFreq=5,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=50,  # The batch size during training.
    valid_batch_size=50,  # The batch size used for validation/test set.
    exchange='AUDJPY',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model="",  # Path to a saved model we want to start from.
    sum_pool = False,
    mom_start = 0.5,
    mom_end = 0.99,
    mom_epoch_interval = 60,
    learning_rate_decay=0.99995,
    #learning_rate_decay=0.98,
    predict=False,
    input_pred=None
):

    '''
        Main function for LSTM training
    '''
    model_path = "/user/j/jgpavez/rnn_trading/models/"
    data_path = "/user/j/jgpavez/rnn_trading/data/"
    log_path = "/user/j/jgpavez/rnn_trading/logs/"


    saveto = exchange + '_model.npz'
    params_file = exchange + '_params.npz'
    dataset = exchange + '_{0}.csv'.format(tick)

    saveto = os.path.join(model_path, saveto)
    params_file = os.path.join(data_path, params_file)

    ydim = 1
    #n_iter = 10
    n_iter = 24
    # Model options
    model_options = locals().copy()

    if predict == True:
        return predict_lstm(input_pred, model_options)


    print "model options", model_options

    print 'Loading data'
    train, valid, test, mean, std = read_data(max_len=n_iter, path=dataset, params_file=params_file,min=(tick=='minute'))

    #YDIM??
    #number of labels (output)

    theano.config.optimizer = 'None'

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params(saveto, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x,
     y, f_pred_prob, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U']**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                             x, y, cost)

    print 'Optimization'


    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size,
                                   shuffle=True)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size,
                                  shuffle=True)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])
    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    mom = 0

    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            if eidx < model_options['mom_epoch_interval']:
                mom = model_options['mom_start']*\
                (1.0 - eidx/model_options['mom_epoch_interval'])\
                  + mom_end*(eidx/model_options['mom_epoch_interval'])
            else:
                mom = mom_end

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray formet.
                # It return something of the shape (minibatch maxlen, n samples)
                x, y = prepare_data(x, y, model_options['n_iter'],model_options['n_input'])

                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue
                n_samples += x.shape[1]
                cost = f_grad_shared(x, y)
                f_update(lrate,mom)

                #decay
                #TODO: CHECK THIS LEARNING RATE
                #lrate = learning_rate_decay*lrate

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    with open(log_path + 'log_{0}_{0}.log'.format(dim_proj, n_iter), 'a') as log_file:
                        log_file.write('Epoch {0} Update {1} Cost {2}\n'.format(eidx, uidx, cost))

            #decay
            #TODO: CHECK THIS LEARNING RATE
            lrate = learning_rate_decay*lrate
            if numpy.mod(eidx, validFreq) == 0:
                use_noise.set_value(0.)
                #train_err = pred_error(f_pred_prob, prepare_data, train, kf, model_options)
                valid_err = pred_error(f_pred_prob, prepare_data, valid, kf_valid, model_options)
                test_err = pred_error(f_pred_prob, prepare_data, test, kf_test, model_options)
                #bckfr_err = backforecast(f_pred_prob, test, model_options)
                #r2_score = R_score(f_pred_prob, prepare_data, test, kf_test, model_options)
                bckfr_err = 0.
                r2_score = 0.

                #history_errs.append([valid_err, test_err])
                history_errs.append([valid_err, bckfr_err])

                if (eidx == 0 or
                    test_err <= numpy.array(history_errs)[:,
                                                           1].min()):
                    #bckfr_err <= numpy.array(history_errs)[:,
                    #                                       1].min()):

                    best_p = unzip(tparams)
                    bad_counter = 0

                with open(log_path + 'log_{0}_{0}.log'.format(dim_proj, n_iter), 'a') as log_file:
                    log_file.write('Valid {0} Test {1}\n'.format(valid_err,test_err))
                print('Valid',valid_err,
                       'Test ', test_err, 
                       'Backfore ', bckfr_err,
                       'R2 score ', r2_score)

                if (len(history_errs) > patience and
                    valid_err >= numpy.array(history_errs)[:-patience,
                                                           0].min()):
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break


            if numpy.mod(eidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print 'Done'


            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    #train_err = pred_error(f_pred_prob, prepare_data, train, kf, model_options)
    valid_err = pred_error(f_pred_prob, prepare_data, valid, kf_valid, model_options)
    test_err = pred_error(f_pred_prob, prepare_data, test, kf_test, model_options)
    #bckfr_err = backforecast(f_pred_prob, test, model_options)
    #r2_score = R_score(f_pred_prob, prepare_data, test, kf_test, model_options)
    r2_score= 0.
    bckfr_err = 0.

    print 'Valid ', valid_err, 'Test ', test_err, 'Backforecasting ', bckfr_err, ' R2 score: ', r2_score

    numpy.savez(saveto, train_err=train_err,
                valid_err=valid_err, test_err=test_err,
                history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':

    # We must have floatX=float32 for this tutorial to work correctly.
    theano.config.floatX = "float32"
    # The next line is the new Theano default. This is a speed up.
    #theano.config.scan.allow_gc = False
    exchange = 'AUDJPY'
    tick = 'hour'
    if len(sys.argv) >= 2:
        exchange = sys.argv[1]
    if len(sys.argv) >= 3:
        tick = sys.argv[2]

    # See function train for all possible parameter and there definition.
    train_lstm(
        #reload_model="lstm_model.npz",
        exchange=exchange,
        max_epochs=20,
        #max_epochs=162,
        tick=tick
    )

