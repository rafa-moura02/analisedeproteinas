import argparse
import os
import csv
import json
import warnings
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering,
    Birch, AffinityPropagation, MeanShift, OPTICS, DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, adjusted_rand_score, normalized_mutual_info_score
)
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
PAIRS = [a + b for a in AA for b in AA]

def read_fasta(path):
    seqs = {}
    cur_id, cur_seq = None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                header = line[1:].strip()
                cur_id = header.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_seq)
    return seqs

def features_2x2_binary(seq, skips, headers_index):
    s = seq.upper()
    n = len(s)
    vec = np.zeros(len(headers_index), dtype=np.float64)
    for x in skips:
        step = x + 1
        if n <= step:
            continue
        for i in range(n - step):
            a = s[i]
            b = s[i + step]
            if a in AA_SET and b in AA_SET:
                key = f"{a}{b}|skip={x}"
                j = headers_index.get(key)
                if j is not None:
                    vec[j] = 1.0
    return vec

def load_labels_csv(path):
    df = pd.read_csv(path)
    need = {"seq_id", "label"}
    if not need.issubset(df.columns):
        raise ValueError("labels_csv deve conter colunas: seq_id,label")
    return dict(zip(df["seq_id"].astype(str), df["label"].astype(str)))

def _best_f1_by_hungarian(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)))
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            ti = (y_true == lt).astype(int)
            pj = (y_pred == lp).astype(int)
            f1_pair = f1_score(ti, pj, zero_division=0)
            cost[i, j] = -f1_pair
    ri, cj = linear_sum_assignment(cost)
    mapping = {labels_pred[j]: labels_true[i] for i, j in zip(ri, cj)}
    y_pred_aligned = np.array([mapping.get(y, y) for y in y_pred])
    f1_macro = f1_score(y_true, y_pred_aligned, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_aligned, average="weighted", zero_division=0)
    return f1_macro, f1_weighted

def _internal_metrics(X, labels):
    n_eff = len(set(labels)) - (1 if -1 in labels else 0)
    if n_eff < 2 or len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels),
    }

def _external_metrics_if_any(labels_true, labels_pred):
    if labels_true is None:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    labels_true = np.asarray(labels_true, dtype=object)
    labels_pred = np.asarray(labels_pred, dtype=object)
    mask = np.array([lt is not None for lt in labels_true])
    if mask.sum() < 2:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    y_t = labels_true[mask]
    y_p = labels_pred[mask]
    if np.unique(y_t).size < 2 or np.unique(y_p).size < 2:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    f1m, f1w = _best_f1_by_hungarian(y_t, y_p)
    ari = adjusted_rand_score(y_t, y_p)
    nmi = normalized_mutual_info_score(y_t, y_p)
    return {"f1_macro": f1m, "f1_weighted": f1w, "ari": ari, "nmi": nmi}

def run_all_clusterings(X, seq_ids, labels_true=None, max_k=12, run_dbscan=False, run_all=False):
    n, d = X.shape
    max_k = max(2, min(max_k, n - 1))
    results = []
    def add_result(algo, params, labels_pred):
        met_int = _internal_metrics(X, labels_pred)
        met_ext = _external_metrics_if_any(labels_true, labels_pred)
        row = {
            "algo": algo,
            "params": json.dumps(params, ensure_ascii=False),
            "n_clusters_found": int(len(set(labels_pred)) - (1 if -1 in labels_pred else 0)),
            **met_int, **met_ext,
        }
        results.append(row)
    K_RANGE = list(range(2, max_k + 1))
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        add_result("KMeans", {"k": k}, km.fit_predict(X))
        mb = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(512, n))
        add_result("MiniBatchKMeans", {"k": k}, mb.fit_predict(X))
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        add_result("Agglomerative(ward)", {"k": k}, agg.fit_predict(X))
        try:
            agg_avg = AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean")
            add_result("Agglomerative(average)", {"k": k}, agg_avg.fit_predict(X))
        except TypeError:
            agg_avg = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean")
            add_result("Agglomerative(average)", {"k": k}, agg_avg.fit_predict(X))
        try:
            spec = SpectralClustering(n_clusters=k, random_state=0, assign_labels="kmeans", n_init=10)
            add_result("Spectral", {"k": k}, spec.fit_predict(X))
        except Exception:
            pass
        bir = Birch(n_clusters=k)
        add_result("Birch", {"k": k}, bir.fit_predict(X))
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=0
            )
            add_result("GaussianMixture", {"k": k, "cov": "diag", "reg": 1e-6},
                       gmm.fit(X).predict(X))
        except Exception:
            add_result("GaussianMixture", {"k": k, "cov": "diag", "reg": 1e-6, "status": "failed"},
                       np.full(X.shape[0], -1, dtype=int))
    if run_all:
        try:
            ap = AffinityPropagation(random_state=0)
            add_result("AffinityPropagation", {}, ap.fit_predict(X))
        except Exception:
            pass
        try:
            ms = MeanShift()
            add_result("MeanShift", {}, ms.fit_predict(X))
        except Exception:
            pass
        try:
            op = OPTICS(min_samples=max(5, int(0.02 * n)))
            add_result("OPTICS", {"min_samples": int(max(5, 0.02 * n))}, op.fit_predict(X))
        except Exception:
            pass
    if run_dbscan:
        X_std = np.std(X, axis=0).mean() + 1e-8
        for eps_mult in [0.5, 1.0, 1.5]:
            eps = eps_mult * X_std
            for ms in [3, 5, 10]:
                try:
                    db = DBSCAN(eps=float(eps), min_samples=ms)
                    add_result("DBSCAN", {"eps": round(float(eps), 6), "min_samples": ms}, db.fit_predict(X))
                except Exception:
                    pass
    return pd.DataFrame(results)

def correlate_internal_with_f1(df):
    out_rows = []
    def _corr_block(sub, tag):
        for metric in ["silhouette", "calinski", "davies"]:
            sub_ok = sub[["f1_macro", metric]].dropna()
            if len(sub_ok) >= 3 and sub_ok["f1_macro"].nunique() > 1 and sub_ok[metric].nunique() > 1:
                p = pearsonr(sub_ok["f1_macro"], sub_ok[metric])[0]
                s = spearmanr(sub_ok["f1_macro"], sub_ok[metric])[0]
            else:
                p, s = np.nan, np.nan
            out_rows.append({"scope": tag, "metric": metric, "pearson": p, "spearman": s, "n": len(sub_ok)})
    _corr_block(df, "GLOBAL")
    for algo, sub in df.groupby("algo"):
        _corr_block(sub, f"ALGO::{algo}")
    return pd.DataFrame(out_rows)

def build_and_predict(algo, params, X):
    if algo == "KMeans":
        k = int(params["k"])
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    if algo == "MiniBatchKMeans":
        k = int(params["k"])
        return MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(512, X.shape[0])).fit_predict(X)
    if algo == "Agglomerative(ward)":
        k = int(params["k"])
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algo == "Agglomerative(average)":
        k = int(params["k"])
        try:
            return AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean").fit_predict(X)
        except TypeError:
            return AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean").fit_predict(X)
    if algo == "Spectral":
        k = int(params["k"])
        return SpectralClustering(n_clusters=k, random_state=0, assign_labels="kmeans", n_init=10).fit_predict(X)
    if algo == "Birch":
        k = int(params["k"])
        return Birch(n_clusters=k).fit_predict(X)
    if algo == "GaussianMixture":
        k = int(params["k"])
        cov = params.get("cov", "diag")
        reg = float(params.get("reg", 1e-6))
        try:
            return GaussianMixture(n_components=k, covariance_type=cov, reg_covar=reg, random_state=0).fit(X).predict(X)
        except Exception:
            return np.full(X.shape[0], -1, dtype=int)
    if algo == "AffinityPropagation":
        return AffinityPropagation(random_state=0).fit_predict(X)
    if algo == "MeanShift":
        return MeanShift().fit_predict(X)
    if algo == "OPTICS":
        ms = int(params.get("min_samples", max(5, int(0.02 * X.shape[0]))))
        return OPTICS(min_samples=ms).fit_predict(X)
    if algo == "DBSCAN":
        eps = float(params["eps"])
        ms = int(params["min_samples"])
        return DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    return np.full(X.shape[0], -1, dtype=int)

def pick_best_configuration(df, select_by="silhouette"):
    df = df.copy()
    df = df[df["n_clusters_found"] >= 2]
    if select_by == "silhouette":
        df["_rank_key"] = list(zip(-df["silhouette"].fillna(-np.inf),
                                   -df["calinski"].fillna(-np.inf),
                                   df["davies"].fillna(np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    if select_by == "calinski":
        df["_rank_key"] = list(zip(-df["calinski"].fillna(-np.inf),
                                   -df["silhouette"].fillna(-np.inf),
                                   df["davies"].fillna(np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    if select_by == "davies":
        df["_rank_key"] = list(zip(df["davies"].fillna(np.inf),
                                   -df["silhouette"].fillna(-np.inf),
                                   -df["calinski"].fillna(-np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    return df.iloc[0].to_dict()

def make_plots(outdir, X_pca, ids, labels_true_map, best_algo, best_params, best_labels, df_var, df_clust):
    import matplotlib.pyplot as plt
    import json, os

    os.makedirs(outdir, exist_ok=True)
    x1, x2 = X_pca[:, 0], X_pca[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    y = df_var["explained_variance_ratio"].values
    axes[0].plot(range(1, len(y) + 1), y, marker="o")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA Scree")

    if labels_true_map:
        labs = [labels_true_map.get(i, None) for i in ids]
        uniq = sorted(list({l for l in labs if l is not None}))
        if len(uniq) >= 2:
            for l in uniq:
                m = [li == l for li in labs]
                axes[1].scatter(x1[m], x2[m], s=16, label=str(l), alpha=0.8)
            axes[1].legend(title="Label")
        else:
            axes[1].scatter(x1, x2, s=16, alpha=0.8)
    else:
        axes[1].scatter(x1, x2, s=16, alpha=0.8)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PCA PC1×PC2")

    sc = axes[2].scatter(x1, x2, c=best_labels, s=16, alpha=0.85)
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title(f"Best ({best_algo}) by clusters")

    df_km = df_clust[df_clust["algo"] == "KMeans"].copy()
    if not df_km.empty and "params" in df_km:
        df_km["k"] = df_km["params"].apply(lambda s: json.loads(s).get("k") if isinstance(s, str) else None)
        df_km = df_km.dropna(subset=["k"]).sort_values("k")
        if df_km["silhouette"].notna().any():
            axes[3].plot(df_km["k"], df_km["silhouette"], marker="o", label="Silhouette")
        if df_km["calinski"].notna().any():
            axes[3].plot(df_km["k"], df_km["calinski"], marker="o", label="Calinski")
        if df_km["davies"].notna().any():
            axes[3].plot(df_km["k"], df_km["davies"], marker="o", label="Davies")
        axes[3].set_xlabel("k")
        axes[3].set_ylabel("score")
        axes[3].set_title("KMeans: métricas internas vs k")
        axes[3].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "painel_completo.png"), dpi=150)
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="2X2 binário + PCA + Clustering (métricas internas/externas).")
    ap.add_argument("--fasta", required=True, help="Caminho do arquivo FASTA")
    ap.add_argument("--outdir", required=True, help="Diretório de saída")
    ap.add_argument("--skips", default="0,1", help="Skips separados por vírgula (ex.: 0,1,2)")
    ap.add_argument("--limit", type=int, default=0, help="Processa apenas as N primeiras sequências")
    ap.add_argument("--pca_components", type=int, default=200, help="Número de componentes PCA")
    ap.add_argument("--max_k", type=int, default=12, help="Máximo de clusters (k)")
    ap.add_argument("--select_by", default="silhouette", choices=["silhouette","calinski","davies"], help="Critério de seleção")
    ap.add_argument("--run_dbscan", action="store_true", help="Executar DBSCAN")
    ap.add_argument("--run_all", action="store_true", help="Executar todos os métodos extras")
    ap.add_argument("--plot", action="store_true", help="Gerar gráficos")
    ap.add_argument("--labels_csv", help="CSV opcional com seq_id,label")
    args = ap.parse_args()

    if os.path.exists(args.outdir):
        for item in os.listdir(args.outdir):
            item_path = os.path.join(args.outdir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception:
                pass
    else:
        os.makedirs(args.outdir, exist_ok=True)

    skips = [int(s.strip()) for s in args.skips.split(",") if s.strip()]
    seqs = read_fasta(args.fasta)
    if args.limit > 0:
        seqs = dict(list(seqs.items())[:args.limit])
    seq_ids = list(seqs.keys())
    headers = [f"{p}|skip={x}" for x in skips for p in PAIRS]
    headers_index = {h: i for i, h in enumerate(headers)}
    X = np.array([features_2x2_binary(seqs[i], skips, headers_index) for i in seq_ids], dtype=np.float64)
    df_feat = pd.DataFrame(X, index=seq_ids, columns=headers)
    pca = PCA(n_components=min(args.pca_components, X.shape[1]-1))
    X_pca = pca.fit_transform(X)
    df_var = pd.DataFrame({"explained_variance_ratio": pca.explained_variance_ratio_})
    labels_true_map = load_labels_csv(args.labels_csv) if args.labels_csv else None
    labels_true = [labels_true_map.get(i) if labels_true_map else None for i in seq_ids]
    df_clust = run_all_clusterings(X_pca, seq_ids, labels_true, args.max_k, args.run_dbscan, args.run_all)
    df_clust.to_csv(os.path.join(args.outdir, "cluster_metrics.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC)
    best = pick_best_configuration(df_clust, args.select_by)
    best_algo = best["algo"]
    best_params = json.loads(best["params"]) if isinstance(best["params"], str) else best["params"]
    best_labels = build_and_predict(best_algo, best_params, X_pca)
    pd.DataFrame({"seq_id": seq_ids, "cluster": best_labels}).to_csv(os.path.join(args.outdir, "best_assignments.csv"), index=False)
    df_corr = correlate_internal_with_f1(df_clust)
    df_corr.to_csv(os.path.join(args.outdir, "correlations.csv"), index=False)
    if args.plot:
        make_plots(args.outdir, X_pca, seq_ids, labels_true_map, best_algo, best_params, best_labels, df_var, df_clust)

if __name__ == "__main__":
    main()