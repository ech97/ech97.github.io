---
title: POSIX Shared Memory
categories:
   - IPC
tags:
   - IPC
   - SharedMemory
---

## 개요

>POSIX의 경우에는 [[Mmap]] 써서 [[File Descriptor]]처럼 사용 가능 (I/O [[📌Multiplexing]]에 활용 가능)
>
>- 실제 파일은 아님
>- Shared Memory 객체 생성 후 [[Mmap]]을 통해 메모리에 올려줘서 사용
>  - SysV에서는 Key를 통해 Shared Memory ID를 얻고, Memory에 Attach

- API
  - shm_open: Shared Memory를 생성하거나 생성되어있는것을 여는 API
    - SHM_NAME 입력 시, 파일 이름이 항상 '/'로 시작해야함
    - mode(permission)의 경우는 O_CREAT을 하는 경우를 제외하고는 0으로 설정해야함
  - shm_unlink: Shared Memory 삭제 (파일이름은 여기서도 '/'로 시작해야함)
  - fstat: 파일의 상태를 읽어오는 함수
    - Shared Memory 파일의 상태를 읽어와 메모리 사이즈를 파악하는 용도로 사용

---
## 구현

```cmake	
# librt.so 필요
ADD_TARGET_LIBRARIES ( ${OUTPUT_ELF} PRIVATE librt.so )
```

```c
#include <unistd.h>	// posix 표준으로 식별되는 구현 특성 정의 (이번 posix shm이랑은 큰 관련 X)
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fnctl.h>
#include <errno.h>
#include <string.h>

#include <sys/mman.h>

#define POSIX_SHM_NAME		"/shared"
#define SHM_SEGMENT_SIZE	4096

void print_usage(char* progname) {
    printf("%s [write | read]\n", progname);
}

int do_read(void) {
    int shm_fd;
    char* shmAddr;
    char buffer[SHM_SEGMENT_SIZE];
    
    if ((shm_fd = shm_open(POSIX_SHM_NAME, O_RDWR, 0666)) == -1) {
        perror("Failed to open shm_fd\n");
        return -1;
    }
    
    shmAddr = (char*)mmap(NULL, SHM_SEGMENT_SIZE, PROT_READ | RPOT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shmAddr == MAP_FAILED) {
        perror("mmap()\n");
        close(shm_fd);
        return -1;
    }
    
    memcpy(buffer, shmAddr, sizeof(buffer));
    printf("from writer: %s\n", buffer);
    
    munmap(shmAddr, SHM_SEGMENT_SIZE);
    close(shm_fd);
    
    shm_unlink(POSIX_SHM_NAME);
    return 0;
}

int do_write(void) {
    int shm_fd;
    char* shmAddr;
    char buffer[SHM_SEGMENT_SIZE];
    
    snprintf(buffer, SHM_SEGMENT_SIZE, "This is writer!!!\n");
    
    if ((shm_fd = shm_open(POSIX_SHM_NAME, O_RDWR | O_CREAT | O_EXCL, 0666)) == -1) {
        perror("Failed to create shm_fd\n");
        return -1;
    }
    
    // 할당될 메모리보다 파일 크기가 작으면 안되니깐 파일 크기 조정
    if (ftruncate(shm_fd, SHM_SEGMENT_SIZE) == -1) {
        perror("ftruncate()\n");
        close(shm_fd);
        return -1;
    }
    
    shmAddr = (char*)mmap(NULL, SHM_SEGMENT_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shmAddr == MAP_FAILED) {
        perror("mmap()\n");
        close(shm_fd);
        return -1;
	}
   
    // 주의: sizeof(포인터) == 8Byte
    memcpy(shmAddr, buffer, sizeof(buffer));
    
    munmap(shmAddr, SHM_SEGMENT_SIZE);
    close(shm_fd);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
    }
    
    if (!strcmp(argv[1], "write")) {
        do_write();
    } else if (!strcmp(argv[1], "read")) {
        do_read();
    } else {
        print_usage(argv[1]);
        return -1;
    }
    
    return 0;
}
```
