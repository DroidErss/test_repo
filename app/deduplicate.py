import pandas as pd
import logging

class Deduplication:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def deduplicate_records(self, df, dedup_config=None) -> pd.DataFrame:
        # ---- extracting config values ----
        if not dedup_config:
            self.logger.info('Config is empty, returning original dataframe')
            return df, pd.DataFrame()
        
        keys = dedup_config.get('keys', [])
        flags = dedup_config.get('flags', {})
        funnel_priority = dedup_config.get('funnel_priority', [])
        threshold_for_removing = dedup_config.get('threshold_for_removing')

        if not keys:
            raise ValueError('keys are empty')
        
        # ---- whitespace removing ----
        df[keys] = df[keys].apply(lambda x: x.astype(str).str.strip())

        # ---- get priority columns ----
        col_apply  = flags.get('APPLY')
        col_admit  = flags.get('ADMIT')
        col_commit = flags.get('COMMIT')
        col_enroll = flags.get('ENROLL')

        # ---- check missing columns ----
        missing_cols = [col for col in [col_apply, col_admit, col_commit, col_enroll] if col and col not in df.columns]
        if missing_cols:
            raise ValueError(f'Dataset has no flag columns: {missing_cols}')

        # ---- creating new stage column ----
        def derive_stage(row):
            if row.get(col_enroll) == 1:
                return 'ENROLL'
            elif row.get(col_commit) == 1:
                return 'COMMIT'
            elif row.get(col_admit) == 1:
                return 'ADMIT'
            elif row.get(col_apply) == 1:
                return 'APPLY'
            else:
                return None

        df['STAGE'] = df.apply(derive_stage, axis=1)

        # ---- defining priority ----
        score_map = {stage: len(funnel_priority) - i for i, stage in enumerate(funnel_priority)}
        df['_score'] = df['STAGE'].map(score_map).fillna(0).astype(int)

        # ---- duplicate group mask ----
        dup_mask = df.duplicated(subset=keys, keep=False)
        if not dup_mask.any():
            df = df.drop(['STAGE', '_score'], axis=1)

            stats = {
                'total_rows': len(df),
                'removed_rows': 0,
                'removed_share': 0.0,
                'threshold': threshold_for_removing,
            }

            self.logger.info('No duplicates found, returning original dataframe')

            return df, pd.DataFrame() # return original dataframe and empty dataframe for log info
        
        keep_idx = df.loc[dup_mask].groupby(keys, dropna=False)['_score'].idxmax()
        flag_cols = [col for col in [col_apply, col_admit, col_commit, col_enroll] if col]

        to_remove_mask = dup_mask & ~df.index.isin(keep_idx)

        # ---- log df formation ----
        removed_log_df = self._log_formation(df, to_remove_mask, keys, flag_cols, keep_idx)

        # ---- final df ----
        df_deduped = df.drop(['STAGE', '_score'], axis=1).loc[~to_remove_mask].copy()

        # ---- logging some additional info ----
        total = len(df)
        removed_n = int(to_remove_mask.sum())
        removed_share = (removed_n / total) if total else 0.0
        threshold = (threshold_for_removing / 100.0) if threshold_for_removing > 1 else float(threshold_for_removing)
        
        stats = {
                'total_rows': total,
                'removed_rows': removed_n,
                'removed_share': removed_share,
                'threshold': threshold,
            }
        
        self.logger.info("Dedup stats: %s", stats)
        
        # ---- warning if more than 2% of the file was removed ----
        if removed_share > threshold:
            self.logger.info(
                f'WARNING: removed {removed_n} of {total} rows ({removed_share:.2%}) due to duplication, exceeds threshold {threshold:.2%}'
                )

        return df_deduped, removed_log_df
    
    def _log_formation(self, df, to_remove_mask, keys, flag_cols, keep_idx) -> pd.DataFrame:
        # ---- log df formation ----
        removed = df.loc[to_remove_mask, keys + flag_cols + ['STAGE', '_score']].copy()
        kept = df.loc[keep_idx, keys + ['STAGE', '_score']].copy()
        removed['_merge_key'] = removed[keys].astype(str).agg('||'.join, axis=1)
        kept['_merge_key'] = kept[keys].astype(str).agg('||'.join, axis=1)
        
        removed = removed.merge(
            kept[['_merge_key', 'STAGE', '_score']].rename(columns={'STAGE': 'STAGE_kept', '_score': '_score_kept'}),
            on='_merge_key', how='left'
        )

        def reason_fn(r):
            if r['_score'] < r['_score_kept']:
                return 'duplicate: lower-priority stage'
            elif r['_score'] == r['_score_kept']:
                return 'duplicate: same stage, kept first'
            return 'duplicate'

        removed['reason'] = removed.apply(reason_fn, axis=1)

        removed_log_df = removed[keys + ['STAGE', 'STAGE_kept', '_score', '_score_kept', 'reason']].copy()

        return removed_log_df