"""Tab-feature pretext CNN: predict the 16 tabular features FROM THE IMAGE (not redshift).

Same trunk + 64-d embedding as photoz_cnn.build_cnn; the head is one 2-Gaussian MDN per
tabular feature (output (16, 6) = [pi(2) | mu(2) | sigma(2)] per feature). Targets are the
baseline's 16 features (fusion.tabular_features), z-scored per feature on the train split
(nan-aware); NaN targets are masked out of the NLL. The point is the EMBEDDING: after
training, feed `embedding` activations (+ tab16) to the HGB head and judge by sigma_MAD.

    from tab_cnn import train_tab
    train_tab(data_dir="/content/data", mlflow_token="<token>")
"""
import os
import inspect
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model

import eval as ev
import fusion as fu
from photoz_cnn import (build_cnn, load_into_ram, resolve_train_index, setup_mlflow,
                        MLFLOW_URI, augment)

N_FEAT, K = 16, 2


# ============================== model ==============================
def build_tab_cnn(input_shape, embed_dim=64, l2=1e-4, drop=0.4, spatial_drop=0.1):
    """photoz trunk -> embedding(64) -> per-feature 2-Gaussian MDN heads -> (16, 6)."""
    base = build_cnn(input_shape, embed_dim=embed_dim, l2=l2, drop=drop, spatial_drop=spatial_drop)
    x = base.get_layer('embedding_drop').output           # (B, 64) post-dropout embedding
    pi = L.Reshape((N_FEAT, K), name='tab_pi_r')(L.Dense(N_FEAT * K, name='tab_pi')(x))
    pi = L.Softmax(axis=-1, name='tab_pi_sm')(pi)
    mu = L.Reshape((N_FEAT, K), name='tab_mu_r')(L.Dense(N_FEAT * K, name='tab_mu')(x))
    sig = L.Reshape((N_FEAT, K), name='tab_sig_r')(L.Dense(N_FEAT * K, name='tab_sig')(x))
    sig = L.Lambda(lambda t: tf.nn.softplus(t) + 1e-3, output_shape=(N_FEAT, K),
                   name='tab_sig_sp')(sig)   # stable, floored sigma; output_shape so load_model works
    # (exponential blows up in the first batches -> 1e10 NLL spikes; softplus+floor is tame)
    out = L.Concatenate(axis=-1, name='tab')([pi, mu, sig])   # (B, 16, 6)
    return Model(base.input, out, name=f'tab_cnn-mdn{K}x{N_FEAT}')


def tab_mdn_nll():
    """Masked NLL of a per-feature K-Gaussian mixture. y_true (B,16) z-scored, NaN = missing;
    y_pred (B,16,6) = [pi(K) | mu(K) | sigma(K)]. Missing targets contribute nothing."""
    def loss(y_true, y_pred):
        pi, mu, sig = y_pred[..., :K], y_pred[..., K:2 * K], y_pred[..., 2 * K:]
        mask = tf.cast(tf.math.is_finite(y_true), tf.float32)          # (B,16)
        y = tf.expand_dims(tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true)), -1)
        log_comp = (tf.math.log(pi + 1e-8) - 0.5 * tf.math.log(2 * np.pi * sig ** 2 + 1e-8)
                    - 0.5 * ((y - mu) / (sig + 1e-8)) ** 2)            # (B,16,K)
        nll = -tf.reduce_logsumexp(log_comp, axis=-1)                  # (B,16)
        return tf.reduce_sum(nll * mask) / (tf.reduce_sum(mask) + 1e-8)
    return loss


# ============================== data ==============================
def tab_targets(data_dir, catalog="catalog_v1.parquet"):
    """(cat, z_all, o2i, Y) where Y (N,16) = the 16 baseline features with NaN for missing
    (raw scale — standardize with train stats before fitting)."""
    cat, z_all, o2i = fu.load_catalog_v4(data_dir, catalog)
    Y, _ = fu.tabular_features(cat)
    return cat, z_all, o2i, Y.astype("float32")


def standardize(Y, train_rows, clip=10.0):
    """Per-feature nan-aware z-score using ONLY train rows, winsorized to +-clip
    (catalog tails reach z-scores of 100+ and spike the NLL). Returns (Y_std, mean, std)."""
    m = np.nanmean(Y[train_rows], axis=0)
    s = np.nanstd(Y[train_rows], axis=0) + 1e-6
    return np.clip((Y - m) / s, -clip, clip), m.astype("float32"), s.astype("float32")


def tab_dataset(X, Y, training=False, batch=256, shuffle_buf=50000, preprocess=None):
    """ram_dataset variant with a (N,16) float target (NaN allowed)."""
    n, H, W = len(X), X.shape[1], X.shape[2]
    ds = tf.data.Dataset.range(n)
    if training:
        ds = ds.shuffle(min(n, shuffle_buf), reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def gather(i):
        xb = tf.numpy_function(lambda ii: X[ii].astype('float16'), [i], tf.float16)
        yb = tf.numpy_function(lambda ii: Y[ii].astype('float32'), [i], tf.float32)
        xb.set_shape([None, H, W, 5]); yb.set_shape([None, N_FEAT])
        return xb, yb

    ds = ds.map(gather, num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess is not None:
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ============================== entry point ==============================
def train_tab(data_dir, crop=24, train_csv=None, N=None, seed=0, es_size=5000,
              batch=256, lr=3e-4, min_lr=1e-5, epochs=50, l2=1e-4, drop=0.4, patience=8,
              preproc='p99', run_name='tab-mdn', mlflow_token=None,
              experiment='tab-cnn', mlflow_uri=MLFLOW_URI):
    """Train the tab-feature pretext CNN. Early-stops on val (masked) NLL.
    Returns (history, model, (feat_mean, feat_std))."""
    from photoz_cnn import make_preprocess
    train_csv = train_csv or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'splits', 'v4-5-train.csv')
    pp_tf = make_preprocess(preproc)
    cat, z_all, o2i, Y = tab_targets(data_dir)
    idx = resolve_train_index(train_csv, data_dir, o2i, N, seed)
    es_idx, train_idx = idx[:es_size], idx[es_size:]
    Y_std, f_mean, f_std = standardize(Y, train_idx)
    print(f'loading {len(train_idx):,} train + {len(es_idx):,} early-stop into RAM (crop={crop}, preproc={preproc})...')
    Xtr, _ = load_into_ram(train_idx, crop, data_dir, z_all)
    Xes, _ = load_into_ram(es_idx, crop, data_dir, z_all)
    print(f'train {Xtr.shape} ({Xtr.nbytes / 1e9:.1f} GB float16) | '
          f'target NaN frac {np.mean(~np.isfinite(Y_std[train_idx])):.2e}')

    model = build_tab_cnn((crop, crop, 5), l2=l2, drop=drop)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss=tab_mdn_nll())
    train_ds = tab_dataset(Xtr, Y_std[train_idx], training=True, batch=batch, preprocess=pp_tf)
    es_ds = tab_dataset(Xes, Y_std[es_idx], training=False, batch=512, preprocess=pp_tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=min_lr),
    ]
    config = dict(crop=crop, batch=batch, lr=lr, epochs=epochs, l2=l2, drop=drop, seed=seed,
                  loss=f'tab_mdn_nll(K={K}x{N_FEAT}, masked)', target='16 tab features (z-scored)',
                  preproc=preproc, augment='rot90+flip', train_csv=str(train_csv),
                  n_train=len(train_idx), params=int(model.count_params()))

    def _fit():
        return model.fit(train_ds, validation_data=es_ds, epochs=epochs, callbacks=callbacks)

    if setup_mlflow(mlflow_token, mlflow_uri, experiment):
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(config)
            mlflow.log_text('\n\n'.join(inspect.getsource(f) for f in
                            (build_tab_cnn, tab_mdn_nll, standardize)), 'recipe.py')
            hist = _fit()
            mlflow.log_metrics({'best_val_nll': float(min(hist.history['val_loss']))})
            np.save(os.path.join(tempfile.gettempdir(), 'tab_feat_scale.npy'),
                    np.stack([f_mean, f_std]))
            mlflow.log_artifact(os.path.join(tempfile.gettempdir(), 'tab_feat_scale.npy'))
            kpath = os.path.join(tempfile.gettempdir(), f'{run_name}.keras')
            model.save(kpath); mlflow.log_artifact(kpath)
    else:
        hist = _fit()

    print('best val NLL:', min(hist.history['val_loss']))
    return hist, model, (f_mean, f_std)
