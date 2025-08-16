import json

def clean_jsonl_file(input_filepath, output_filepath):
    """
    JSONL 파일에서 유효하지 않은 JSON 라인을 제거하고 결과를 새 파일에 저장합니다.

    Args:
        input_filepath (str): 원본 JSONL 파일 경로
        output_filepath (str): 정리된 내용이 저장될 파일 경로
    """
    total_lines = 0
    valid_lines = 0
    invalid_lines = 0

    print(f"'{input_filepath}' 파일 처리를 시작합니다...")

    # 원본 파일을 읽고, 새로운 파일을 쓰기 모드로 엽니다.
    # 한글 처리를 위해 encoding='utf-8'을 사용합니다.
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            try:
                # 각 라인이 유효한 JSON인지 파싱을 시도합니다.
                json.loads(line)
                
                # 파싱에 성공하면, 원본 라인을 그대로 새 파일에 씁니다.
                outfile.write(line)
                valid_lines += 1
            except json.JSONDecodeError:
                # JSON 파싱에 실패하면, 해당 라인은 건너뜁니다.
                invalid_lines += 1
                print(f"  - {total_lines}번째 라인에서 비정상 JSON을 발견하여 제외합니다.")

    print("\n파일 처리가 완료되었습니다.")
    print(f"  - 총 라인 수: {total_lines}")
    print(f"  - 유효한 JSON 라인 수: {valid_lines}")
    print(f"  - 제거된 비정상 라인 수: {invalid_lines}")
    print(f"정리된 데이터가 '{output_filepath}' 파일에 저장되었습니다.")

# --- 실행 부분 ---
# 처리할 원본 파일 이름을 지정합니다.
input_file = '/workspace/2025-AI-Challeng-finance/generated_finetuning_dataset_from_paper.jsonl'

# 저장할 새로운 파일 이름을 지정합니다.
output_file = 'paper_generate_mid_정보보호산업.jsonl'

# 함수를 실행하여 파일을 정제합니다.
clean_jsonl_file(input_file, output_file)