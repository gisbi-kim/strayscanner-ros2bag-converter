#!/bin/bash

# 원본 디렉토리와 청크 디렉토리 지정
source_dir="images-rotated"
chunk_dir="images-rotated_chunks"
images_per_chunk=30  # 한 청크에 포함되는 이미지 수
k_frames=3  # 스킵할 프레임 수

# 원본 디렉토리에서 jpg 파일 목록을 가져오고 정렬
images=($(ls "$source_dir"/*.jpg | sort))

# 총 이미지 수 계산
total_images=${#images[@]}

# 청크 디렉토리 생성
mkdir -p "$chunk_dir"

# 스킵된 이미지 리스트 생성 및 인덱스 매핑
skipped_images=()
original_indices=()
index=0
for ((i=0; i<total_images; i+=$((k_frames + 1)))); do
    skipped_images+=("${images[i]}")
    original_indices+=($i)
    index=$((index + 1))
done

# 스킵된 이미지 수 계산
total_skipped_images=${#skipped_images[@]}

# 청크 인덱스 초기화
chunk_index=0

for ((i=0; i<total_skipped_images; i+=$images_per_chunk)); do
    # 현재 청크의 끝 인덱스 계산
    end=$(( i + images_per_chunk - 1 ))
    if [ $end -ge $total_skipped_images ]; then
        end=$(( total_skipped_images - 1 ))
    fi

    # 원본 이미지 인덱스에 맞춰 폴더 이름 설정
    start_original_index=${original_indices[i]}
    end_original_index=${original_indices[end]}
    chunk_name="chunk_from_${start_original_index}_to_${end_original_index}"
    if [ "$start_original_index" -eq 0 ]; then
        chunk_name="chunk_from_0_to_${end_original_index}"
    fi
    mkdir -p "$chunk_dir/$chunk_name"

    # 파일 복사
    for ((j=i; j<=end; j++)); do
        cp "${skipped_images[j]}" "$chunk_dir/$chunk_name"
    done
done

echo "Images have been skipped by $k_frames frames, then split into chunks with $images_per_chunk images each, and folder names reflect original indices, copied to $chunk_dir"