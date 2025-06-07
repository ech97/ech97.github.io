---
title: Anonymous Mmap
categories:
   - IPC
tags:
   - IPC
   - SharedMemory
---

## 개요

> 프로세스에 익명의 파일을 가리키는 공간 할당
>
> - File I/O가 없어 [[Mmap]]보다 빠름
> 	- 바로 같은 메모리 영역을 공유하며 Load/Store 오버헤드 감소
> - 서로 **관련있는 (related) 프로세스간만** Sharing 가능
> 	- malloc()을 사용하면 fork()시 프로세스 영역이 공유되지못하고 각자의 영역을 가지는데
> 	- **Anonymous [[Mmap]]으로 할당한 공간은 fork()시 할당된 영역 Sharing 가능**

---
## 구현

```c
#include <string.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SHARED_FILENAME "shared"

struct login_info {
    int pid;
    int counter;
};

static int do_monitor(struct login_info* info) {
    struct login_info local;	// 값 변경을 파악하기 위한 local 구조체
    
    while(1 == 1) {
        // memcmp를 통해 size만큼 포인터가 가르키는곳의 데이터가 같은지 확인
        if (memcmp(&local, info, sizeof(struct login_info))) {
            printf("pid = %d, count %d\n", info->pid, info->counter);
            memcpy(&local, info, sizeof(struct login_info));
        }
        return 0;
    }
}

static int do_changer(struct login_info* info) {
    while(1 == 1) {
        info->pid = getpid();
        info->counter++;
        sleep(1);
    }
    munmap(info, sizeof(struct login_info));
    return 0;
}

int main(int argc, char** argv) {
	struct login_info* info;

    int wstatus;		// child 프로세스의 종료를 알려줄 함수
    int pid;
    
    // 매핑할 메모리 시작 주소, 할당할 크기 (파일 크기), 권한, flag, fd(Anony일땐 -1), offset(파일을 어디서부터 읽을지)
    info = mmap(NULL, sizeof(struct login_info), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    // 여기서 mmap의 경우 메모리 접근 권한도 같이 fork(복사) 됨
    // 따라서 related process에서 같은 memory 영역에 접근 가능!
    pid = fork();
    if (pid == 0) {
        printf("do_changer() started!\n");
        do_changer(info);
    } else if (pid > 0) {
        printf("do_monitor() started!\n");
        do_monitor(info);
    } else {
        perror("fork()");
        return -1;
    }
    return 0;
}
```