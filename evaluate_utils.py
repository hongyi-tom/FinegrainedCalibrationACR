import re
import json
import subprocess
import Levenshtein
import pandas as pd

def process_generated_code(input_text):
    """
    Cleans the generated code revision by removing formatting markers and special tokens. 
    Args:
        input_text (str): The generated code revision string, potentially containing fences, markers, and end tokens.
    Returns:
        str: The cleaned code revision string with all special markers and extra tokens removed.
    """
    # Remove markdown code block markers (with any language)
    processed = re.sub(r'^```[a-zA-Z0-9_+-]*\n?', '', input_text)
    processed = re.sub(r'```\w*\s*', '', processed)
    processed = re.sub(r'```\s*', '', processed)
    
    # Remove special markers
    processed = processed.split("[_END_FIXED_CODE]")[0]
    processed = re.sub(r'\[_BEGIN_FIXED_CODE\]\s*', '', processed)
    processed = re.sub(r'\[_END_FIXED_CODE\]\s*', '', processed)
    processed = re.sub(r'<\|im_end\|>\s*', '', processed)
    processed = re.sub(r'<\|endoftext\|>\s*', '', processed)
    processed = re.sub(r'</s>\s*', '', processed)
    processed = re.sub(r'<\|EOT\|>\s*', '', processed)
    processed = re.sub(r'<｜end▁of▁sentence｜>\s*', '', processed)
    
    return processed


def exact_match(s1, s2):
    """
    Compares generated and ground truth code revision strings for exact match after removing all whitespace characters.
    Args:
        s1 (str): The ground truth code revision string.
        s2 (str): The generated code revision string.
    Returns:
        bool: True if the strings are identical ignoring whitespace, False otherwise.
    """
    clean1 = re.sub(r'\s+', '', s1)
    clean2 = re.sub(r'\s+', '', s2)
    return clean1 == clean2


def record_exact_match(golds, generated):
    """
    Computes exact match (EM) scores for a complete set of generated and ground truth code revision strings.
    Args:
        golds (pd.DataFrame): A DataFrame containing the ground truth code revision strings.
        generated (pd.DataFrame): A DataFrame containing generated code revision strings.
    Returns:
        List[bool]: A list of boolean values indicating whether exact match was achieved between 
                    generated and ground truth code revision strings.
    """
    EM_records = []

    for row in range(len(generated)):
        gt = golds.iloc[row]
        preds = process_generated_code(generated.iloc[row].generated_sequence)
        EM_records.append(exact_match(gt, preds))

    print("EXACT MATCH (%): ", round((sum(EM_records)/len(generated))*100,1))
    return EM_records


def create_files(generated):
    """
    Writes each generated code revision string for DCF-Vul into a separate JavaScript file.
    Args:
        generated (pd.DataFrame): An DataFrame containing generated code revision strings for DCF-Vul.
    """
    count = 0
    for example in generated.generated_sequence:
        code = process_generated_code(example)
        output_dir = "static_check/test_" + str(count) + ".js"

        with open(output_dir, "w") as file:
            file.write(code)

        count += 1


def run_static_check(org_id, dir):
    """
    Runs Snyk static code analysis on a set of generated code revision strings for DCF-Vul.
    Args:
        org_id (str): The organization ID used for the Snyk scan.
        dir (str): The directory path containing the generated code revisions to be scanned.
    """
    try:
        result = subprocess.run('snyk code test --org={org_id} --json-file-output=report.json', 
                                cwd=dir, 
                                shell=True, 
                                capture_output=True, 
                                text=True)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Stderr: {e.stderr}")


def collect_static_results(dir, generated):
    """
    Processes the Snyk static analysis JSON report and determines which code revisions generated for DCF-Vul are plausible.
    Args:
        dir (str): Path to the JSON report file produced by the static analysis.
        generated (list): List of generated generated code revisions to evaluate against the static analysis results.
    Returns:
        list[bool]: A list of booleans indicating plausibility for each generated code revision.
                    (True if no static alarms, False otherwise).
    """
    with open(dir, 'r') as f:
     data = json.load(f)

    report = pd.DataFrame(data['runs'][0]['results'])

    static_alarms = []
    plausible_records = []

    for alarm in report.locations:
        loc = int(alarm[0]['physicalLocation']['artifactLocation']['uri'].split("_")[1].split('.')[0])
        static_alarms.append(loc)

    for index in range(len(generated)):
        if index in static_alarms:
            plausible_records.append(False)
        else:
            plausible_records.append(True)

    print("PLAUSIBLE (%): ", round((sum(plausible_records)/len(generated))*100,1))

    # Cleanup temporary directory of JavaScript files
    try:
        result = subprocess.run('rm *', 
                                cwd='./static_check/', 
                                shell=True, 
                                capture_output=True, 
                                text=True)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Stderr: {e.stderr}")

    return plausible_records


def security_checks():
    """
    Returns:
        list[str]: A list of identifiers for Snyk static analysis security vulnerability rules
    """
    security_checks = [
        'DisablePoweredBy',
        'ElectronInsecureWebPreferences',
        'ElectronLoadInsecureContent',
        'HardcodedNonCryptoSecret',
        'HardcodedSecret',
        'InsecureCipherNoIntegrity',
        'InsecureECB',
        'InsecureHash',
        'InsecureTLSConfig',
        'InsufficientPostmessageValidation',
        'IntrospectionEnabled',
        'LimitGraphqlDepth',
        'LoopDOS',
        'NoCryptoTimingAttacks',
        'NoHardcodedCredentials',
        'NoHardcodedPasswords',
        'NodeBufferNoOffset',
        'TooPermissiveCorsHeader',
        'TooPermissiveCorsPostMessage',
        'TooSmallRsaKeySizeUsed',
        'UseCsurfForExpress',
        'UseHelmetForExpress',
        'UseSecureWebsockets',
        'WebCookieHttpOnlyDisabledByDefault',
        'WebCookieHttpOnlyDisabledExplicitly',
        'WebCookieSecureDisabledByDefault',
        'WebCookieSecureDisabledExplicitly',
        'CodeInjection',
        'CommandInjection',
        'DOMXSS',
        'FormatString',
        'HTTPSourceWithUncheckedType',
        'IndirectCommandInjection',
        'NoRateLimitingForExpensiveWebOperation',
        'NoRateLimitingForLogin',
        'NoSqli',
        'OR',
        'PT',
        'PrototypePollution',
        'ServerLeak',
        'Sqli',
        'XSS',
        'reDOS'
    ]
    return security_checks


def edit_progress(inputs, golds, generated):
    """
    Compares generated and ground truth code revision strings for token-level edit progress.
    Args:
        inputs (list[str]): List of original code snippet strings before code revision.
        golds (list[str]): List of ground truth code revision strings.
        generated (list[str]): List of generated code revision strings.
    Returns:
        list[bool]: A list of booleans indicating whether each generated code revision shows edit progress.
                    (True if closer to gold than source, False otherwise).
    """
    preds = []
    for row in generated:
        preds.append(process_generated_code(row))

    edit_distance_pred2gold = []
    edit_distance_src2gold = []
    
    for i in range(len(golds)):
        edit_distance_pred2gold.append(
            Levenshtein.distance(re.sub(r'\s+', '', golds[i]), re.sub(r'\s+', '', preds[i]))
        )

        edit_distance_src2gold.append(
            Levenshtein.distance(re.sub(r'\s+', '', golds[i]), re.sub(r'\s+', '', inputs[i]))
        )

    progress = []
    for i in range(len(edit_distance_pred2gold)):
        pred_ = edit_distance_pred2gold[i]
        src_  = max(edit_distance_src2gold[i], 0.01)

        p_ = round( (abs(src_)-abs(pred_) )/abs(src_), 3)
        progress.append(p_)

    binary_progress = [x > 0 for x in progress]

    print("EDIT PROGRESS (%): ", round((sum(binary_progress)/len(generated))*100,1))
    return binary_progress