import os
import re
from pathlib import Path

def extract_title_from_filename(filename):
    """파일명에서 날짜를 제거하고 title을 추출"""
    # YYYY-MM-DD- 패턴 제거
    title = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', filename)
    # .md 확장자 제거
    title = title.replace('.md', '')
    # 하이픈을 언더스코어로 변경
    title = title.replace('-', '_')
    return title

def get_directory_name(file_path):
    """파일이 속한 디렉토리 이름 반환"""
    parent_dir = os.path.dirname(file_path)
    if parent_dir == '' or parent_dir == '.':
        return None
    return os.path.basename(parent_dir)

def has_metadata(content):
    """파일이 이미 메타데이터를 가지고 있는지 확인"""
    return content.strip().startswith('---')

def extract_metadata_and_content(content):
    """메타데이터와 본문을 분리"""
    lines = content.split('\n')
    if not lines[0].strip() == '---':
        return None, content
    
    metadata_end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            metadata_end = i
            break
    
    if metadata_end == -1:
        return None, content
    
    metadata_lines = lines[1:metadata_end]
    content_lines = lines[metadata_end + 1:]
    
    return metadata_lines, '\n'.join(content_lines)

def update_metadata_categories_tags(metadata_lines, category):
    """메타데이터의 categories와 tags를 업데이트"""
    updated_lines = []
    skip_until_next_field = False
    
    for line in metadata_lines:
        if line.startswith('categories:'):
            updated_lines.append('categories:')
            if category:
                updated_lines.append(f'   - {category}')
            else:
                updated_lines.append('   - ')
            skip_until_next_field = True
        elif line.startswith('tags:'):
            updated_lines.append('tags:')
            if category:
                updated_lines.append(f'   - {category}')
            else:
                updated_lines.append('   - ')
            skip_until_next_field = True
        elif line.startswith((' ', '\t')) and skip_until_next_field:
            # categories나 tags의 하위 항목들은 건너뛰기
            continue
        else:
            # 다른 필드 시작
            skip_until_next_field = False
            updated_lines.append(line)
    
    return updated_lines

def remove_toc(content):
    """[toc] 문자열과 목차 섹션 제거"""
    # [toc] 문자열 제거
    content = re.sub(r'\[toc\]', '', content, flags=re.IGNORECASE)
    
    # ## 목차\n--- 패턴 제거 (공백 있는 경우와 없는 경우 모두)
    content = re.sub(r'^## 목차\s*\n---\s*\n', '', content, flags=re.MULTILINE)
    
    return content

def create_metadata(title, category=None):
    """메타데이터 생성"""
    metadata = ["---", f"title: {title}"]
    
    if category:
        metadata.extend([
            "categories:",
            f"   - {category}",
            "tags:",
            f"   - {category}"
        ])
    else:
        metadata.extend([
            "categories:",
            "   - ",
            "tags:",
            "   - "
        ])
    
    metadata.append("---")
    return '\n'.join(metadata) + '\n\n'

def process_markdown_file(file_path):
    """마크다운 파일 처리"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # [toc]와 목차 섹션 제거
        content = remove_toc(content)
        
        # 디렉토리 이름 추출
        category = get_directory_name(file_path)
        
        if has_metadata(content):
            # 메타데이터가 있는 경우 - categories와 tags만 업데이트
            metadata_lines, body_content = extract_metadata_and_content(content)
            if metadata_lines is not None:
                updated_metadata = update_metadata_categories_tags(metadata_lines, category)
                new_content = '---\n' + '\n'.join(updated_metadata) + '\n---\n' + body_content
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"메타데이터 업데이트: {file_path}")
                print(f"  - Category: {category}")
            else:
                print(f"메타데이터 파싱 오류: {file_path}")
        else:
            # 메타데이터가 없는 경우 - 새로 생성
            filename = os.path.basename(file_path)
            title = extract_title_from_filename(filename)
            
            metadata = create_metadata(title, category)
            new_content = metadata + content
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"메타데이터 생성: {file_path}")
            print(f"  - Title: {title}")
            print(f"  - Category: {category}")
        
    except Exception as e:
        print(f"오류 발생 ({file_path}): {e}")

def main():
    """메인 함수"""
    current_dir = Path('.')
    
    # 현재 경로 하위의 모든 .md 파일 찾기
    md_files = list(current_dir.rglob('*.md'))
    
    if not md_files:
        print("처리할 .md 파일이 없습니다.")
        return
    
    print(f"총 {len(md_files)}개의 .md 파일을 찾았습니다.")
    print("처리를 시작합니다...\n")
    
    for file_path in md_files:
        process_markdown_file(file_path)
    
    print("\n모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()