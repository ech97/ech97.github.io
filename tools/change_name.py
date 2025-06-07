import os
import datetime
from pathlib import Path

def rename_files_with_creation_date():
    """현재 경로의 모든 파일 이름 앞에 생성시간을 붙이는 함수"""
    
    current_path = Path.cwd()
    print(f"현재 경로: {current_path}")
    
    # 현재 디렉토리의 모든 파일 찾기 (하위 디렉토리 포함)
    files_to_rename = []
    
    for file_path in current_path.rglob('*.md'):
        if file_path.is_file():
            files_to_rename.append(file_path)
    
    if not files_to_rename:
        print("변경할 파일이 없습니다.")
        return
    
    print(f"총 {len(files_to_rename)}개의 파일을 찾았습니다.")
    
    # 사용자 확인
    response = input("파일 이름을 변경하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업을 취소했습니다.")
        return
    
    renamed_count = 0
    failed_count = 0
    
    for file_path in files_to_rename:
        try:
            # 파일 생성시간 가져오기 (Windows에서는 st_ctime이 생성시간)
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.datetime.fromtimestamp(creation_time)
            
            # 날짜 형식을 YYYY-MM-DD로 포맷
            date_prefix = creation_date.strftime("%Y-%m-%d")
            
            # 기존 파일명
            original_name = file_path.name
            
            # 이미 날짜가 붙어있는지 확인 (YYYY-MM-DD 패턴)
            if original_name.startswith(date_prefix):
                print(f"건너뜀: {original_name} (이미 날짜가 붙어있음)")
                continue
            
            # 파일명의 띄어쓰기를 dash로 변경
            cleaned_name = original_name.replace(' ', '-')
            
            # 새 파일명 생성
            new_name = f"{date_prefix}-{cleaned_name}"
            new_path = file_path.parent / new_name
            
            # 파일명 변경
            file_path.rename(new_path)
            print(f"변경: {original_name} → {new_name}")
            renamed_count += 1
            
        except Exception as e:
            print(f"오류 발생 ({file_path.name}): {e}")
            failed_count += 1
    
    print(f"\n작업 완료!")
    print(f"성공: {renamed_count}개")
    print(f"실패: {failed_count}개")

def preview_changes():
    """변경될 파일명을 미리 보여주는 함수"""
    
    current_path = Path.cwd()
    print(f"현재 경로: {current_path}")
    
    files_to_rename = []
    for file_path in current_path.rglob('*.md'):
        if file_path.is_file():
            files_to_rename.append(file_path)
    
    if not files_to_rename:
        print("변경할 파일이 없습니다.")
        return
    
    print(f"\n변경 예정 파일 목록 (총 {len(files_to_rename)}개):")
    print("-" * 80)
    
    for file_path in files_to_rename[:10]:  # 처음 10개만 미리보기
        try:
            creation_time = os.path.getctime(file_path)
            creation_date = datetime.datetime.fromtimestamp(creation_time)
            date_prefix = creation_date.strftime("%Y-%m-%d")
            
            original_name = file_path.name
            
            if original_name.startswith(date_prefix):
                print(f"건너뜀: {original_name}")
            else:
                cleaned_name = original_name.replace(' ', '-')
                new_name = f"{date_prefix}-{cleaned_name}"
                print(f"{original_name} → {new_name}")
                
        except Exception as e:
            print(f"오류: {file_path.name} - {e}")
    
    if len(files_to_rename) > 10:
        print(f"... 및 {len(files_to_rename) - 10}개 더")

if __name__ == "__main__":
    print("파일 생성시간 이름 변경 도구 (띄어쓰기 → dash 변환 포함)")
    print("=" * 40)
    
    while True:
        print("\n옵션을 선택하세요:")
        print("1. 변경 예정 파일 미리보기")
        print("2. 파일 이름 변경 실행")
        print("3. 종료")
        
        choice = input("선택 (1-3): ")
        
        if choice == "1":
            preview_changes()
        elif choice == "2":
            rename_files_with_creation_date()
        elif choice == "3":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1, 2, 3 중에서 선택해주세요.")