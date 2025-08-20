from app.deduplicate import Deduplication
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)

def main():
    df = input_df

    deduplicator = Deduplication()

    # ---- define config ----
    config = {
        'keys': ['STUDENTID'],
        'flags': {
            'APPLY': 'FLAG_APP', 
            'ADMIT': 'FLAG_ADM', 
            'COMMIT': 'FLAG_CON', 
            'ENROLL': 'FLAG_ENR'
            }, # FLAG_ENR > FLAG_CON > FLAG_ADM > FLAG_APP
        'funnel_priority': ['ENROLL', 'COMMIT', 'ADMIT', 'APPLY'], # ENROLL > COMMIT > ADMIT > APPLY
        'threshold_for_removing': 2 # 2%
    }

    df_deduped, removed_log_df = deduplicator.deduplicate_records(df, config)

if __name__ == "__main__":
    main()