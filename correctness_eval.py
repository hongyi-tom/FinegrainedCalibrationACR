import os
import sys
import warnings
import pandas as pd
import utils.eval_utils as u

# Suppress Warnings
sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore")

# CAUTION DEEPCODE-TRAIN DATA CAN ONLY BE RAN ONCE, THE ORDER WILL RANDOMISE
def main():
    # Read and load inference results from command-line arguments
    result = sys.argv[1]
    benchmark = result.split("_")[-2] + "_" + result.split("_")[-1]
    generated = pd.read_json(benchmark + "/" + result + ".jsonl", lines=True)

    # Read Snyk organisation ID from command-line arguments
    snyk_org_id = sys.argv[2]

    # Read and load relevant benchmark
    if result.split("_")[-1] == 'train':
        dataset = pd.read_parquet(benchmark + "/" + benchmark + ".parquet").reset_index()

        if result.split("_")[-2] == 'deepcode':
            generated['rule'] = dataset['rule']
            generated['input_code'] = dataset['pre_reduced']
            generated['gt'] = dataset['post_reduced']

            bug_generated = generated.loc[~generated['rule'].isin(u.security_checks())]
            bug_generated = bug_generated.reset_index()
            vul_generated = generated.loc[generated['rule'].isin(u.security_checks())]
            vul_generated = vul_generated.reset_index()
        else:
            generated['input_code'] = dataset['pre_review_code']
            generated['gt'] = dataset['post_review_code']  

    else:
        dataset = pd.read_parquet(benchmark + "/" + benchmark + "_test" + ".parquet").reset_index()

        if benchmark == "codereviewqa_trans":
            generated['input_code'] = dataset['pre_review_code']
            generated['gt'] = dataset['post_review_code']  
        else:
            generated['input_code'] = dataset['pre_reduced']
            generated['gt'] = dataset['post_reduced']

    # Evaluate correctness based on the relevant metrics
    if benchmark == "deepcode_train":
        print("BUG")
        bug_generated['PLAUSIBLE'] = False
        bug_generated['EM'] = u.record_exact_match(bug_generated['gt'], bug_generated)
        bug_generated['EP'] = u.edit_progress(bug_generated['input_code'], bug_generated['gt'], bug_generated['generated_sequence'])
        bug_generated['Type'] = 'bug'

        print("-"*150)
        print("VULNERABILITY")
        u.create_files(vul_generated)
        u.run_static_check(snyk_org_id, "./static_check/")
        vul_generated['PLAUSIBLE'] = u.collect_static_results('static_check/report.json', vul_generated)
        vul_generated['EM'] = u.record_exact_match(vul_generated['gt'], vul_generated)
        vul_generated['EP'] = u.edit_progress(vul_generated['input_code'], vul_generated['gt'], vul_generated['generated_sequence'])
        vul_generated['Type'] = 'vul'
        generated = pd.concat([bug_generated, vul_generated], axis=0)

    else:
        if result.split("_")[-1] == "vul":
            u.create_files(generated)
            u.run_static_check(snyk_org_id, "./static_check/")
            generated['PLAUSIBLE'] = u.collect_static_results('static_check/report.json', generated)
        
        generated['EM'] = u.record_exact_match(generated['gt'], generated)
        generated['EP'] = u.edit_progress(generated['input_code'], generated['gt'], generated['generated_sequence'])
        
    # Save Results
    generated.to_json(benchmark + "/" + result + ".jsonl", orient='records', lines=True)


if __name__ == "__main__":
    main()