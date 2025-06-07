---
title: SysV Shared Memory
categories:
   - SharedMemory
tags:
   - SharedMemory
---

## 개요

> id가 있어 서로 관련이 없는 (unrelated) 프로세스간 메모리 공유가 가능하며, File I/O가 없어 빠름
>
> - 프로세스 여러개가 접근하므로, 서로 동기화 작업을 해주는것도 필요 (추후 Semaphore에서 확인예정)

- API
  - shmget: Key와 Size를 넣어 Shared Memory ID를 리턴
    - Shared Memory ID는 [[File Descriptor]]가 아니므로, [[Epoll]], [[Select]] 등 I/O [[📌Multiplexing]] 불가

  - shmat: 생성한 Shared Memory ID를 통해 Shared Memory에 **At**tach한 다음 메모리 주소를 받아옴 ([[Mmap]]과 유사)

  - shmdt: Shared Memory 주소를 통해 **De**tach 하는 API
  - shmctl: Buffer를 통해 커널에서 Shared memory의 정보을 가져오거나 속성 기록
    - 주로 IPC_RMID 명령으로 Shared Memory ID 제거

---
## 구현

```C
#include <stdint.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <unistd.h>

#define IPC_KEY_FILENAME	"/proc"
#define IPC_KEY_PROJ_ID		'a'
#define MEM_SIZE 4096

int32_t shmid;

static void print_usage(char* progname) {
    printf("%s [write | read]\n");
}

static int32_t SharedMemoryFree(void) {
    if (shmctl(shmid, IPC_RMID, 0) == -1) {
        perror("Shmctl()\n");
        return -1;
    }
    printf("Shared Memory end\n");
    return 0;
}

static int32_t SharedMemoryCreate() {
    key_t key;    
    key = ftok(IPC_KEY_FILENAME, IPC_KEY_PROJ_ID);	// 파일이름과 프로젝트 id를 이용해 key 생성 (SysV에선 unique 보장 X)
    if (key == -1) {
        perror("ftok()\n");
        return -1;
    }
    
    // IPC_EXCL: 이미 key에 해당하는 shmid가 있다는 flag (독점적으로 사용하고자 할 때 사용)
    if ((shmid = shmget(key, MEM_SIZE, IPC_CREAT | IPC_EXCL | 0666)) == -1) {
        printf("Maybe Shared Memory already exists\n");
        // 이미 있는 shmid를 제거하기위해 shmid 가져오기
        if ((shmid = shmget(key, 0, 0)) == -1) {
            perror("Shared Memory get failed\n");
            return -1;
		}
        SharedMemoryFree();
        // 기존 key 할당된 shmid를 제거했으니 다시 shmid를 받아와야함
        if ((shmid = shmget(key, MEM_SIZE, IPC_CREAT | 0666)) == -1) {
            perror("Shared Memory Create failed\n");
            return -1;
        }
    }
    printf("Create New Shared Memory ID\n");
    return 0;
}

static int32_t SharedMemoryOpen(void) {
    key_t key;    
    key = ftok(IPC_KEY_FILENAME, IPC_KEY_PROJ_ID);	// 파일이름과 프로젝트 id를 이용해 key 생성 (SysV에선 unique 보장 X)
    if (key == -1) {
        perror("ftok()\n");
        return -1;
    }

    if ((shmid = shmget(key, 0, 0)) == -1) {
        perror("Shared Memory open failed\n");
        return -1;
    }    
    return 0;
}

static int32_t SharedMemoryWrite(char* sharedData, int32_t size) {
    void* shmAddr;
    
    if (size > MEM_SIZE) {
        perror("Shared Memory size over\n");
        return -1;
    }
    
    if ((shmAddr = shmat(shmid, (void*)0, 0)) == (void*)-1) {
        perror("Shmat failed\n");
        return -1;
    }
    
    memcpy((char*)shmAddr, sharedData, size);
    
    printf("Copy to Shared Memory\n");
    if (shmdt(shmAddr) == -1) {
        perror("Shmdt failed\n");
        return -1;
    }

    return 0;
}

static int32_t SharedMemoryRead(char* outBuf) {
    void* shmAddr;
    
    if ((shmAddr = shmat(shmid, (void*)0, 0)) == (void*)-1) {
        perror("Shmat failed\n");
        return -1;
    }
    
    memcpy(outBuf, (char*)shmAddr, sizeof(outBuf));
    
    if (shmdt(shmAddr) == -1) {
        perror("Shmdt failed");
        return -1;
    }
    
    return 0;
}

static int32_t do_write(void) {

    char buffer[MEM_SIZE] = {1, };
    if (SharedMemoryCreate() == -1) {
        perror("SharedMemoryCreate()\n");
        return -1;
    }
    if (SharedMemoryWrite(buffer, sizeof(buffer)) == -1) {
        perror("SharedMemoryWrite()\n");
        return -1;
    }
    sleep(5);
    if (SharedMemoryFree() == -1) {
        perror("SharedMemoryFree()\n");
        return -1;
    }
    return 0;
}

static int32_t do_read(void) {
    char buffer[MEM_SIZE] = {0, };

    if (SharedMemoryOpen() == -1) {
        perror("SharedMemoryOpen()\n");
        return -1;
    }

    while (1 == 1) {
        if (SharedMemoryRead(buffer) == -1) {
            perror("SharedMemoryRead()\n");
            return -1;
        }
        if (buffer[0] == 1) {
            printf("Receive data from Shared Memory!\n");
            break;
        }
    }

    return 0;
}

int32_t main(int32_t argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }
    if (!strcmp(argv[1], "write")) {
        do_write();
    } else if (!strcmp(argv[1], "read")) {
        do_read();
    } else {
        print_usage(argv[0]);
        return -1;
    }

    return 0;
}
```
