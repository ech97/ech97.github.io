---
title: Message Queue (SysV)
categories:
   - IPC
tags:
   - IPC
   - DataTransfer
---

## 개요

> Message 단위 전송

> 하나의 메세지는 하나의 덩어리처럼 처리되며 FIFO를 이용한 Message Queue
> - 메세지 큐 고유의 int형 Key값을 가지며, [[File Descriptor]]와는 다른 개념
>   - 서로 다른 프로세스간 통신 가능
>   - [[File Descriptor]]를 이용하는 [[Select]]와 같은 I/O [[📌Multiplexing]] 도구 사용 **불가**
> - Type을 통해 **하나의 큐에서 여러 프로세스**가 메세지를 골라 가져갈 수 있음 (SysV에서만 지원)

---
## 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ipc.h>

#define IPC_KEY_FILENAME	"/proc"
#define IPC_KEY_PROJ_ID		'a'

static void print_usage(const char *prog) {
    printf("%s (send|recv) (mtype)\n");
}

struct msgbuf {
    long mtype;
#define MDATA_SIZE	64
    char mdata[MDATA_SIZE];
};

static int init_msgq(void) {
    int msgq;
    key_t key;
    struct msgbuf mbuf;
    
    key = ftok(IPC_KEY_FILENAME, IPC_KEY_PROJ_ID);	// 파일이름과 프로젝트 id를 이용해 key 생성 (SysV에선 unique 보장 X)
    if (key == -1) {
        perror("ftok()\n");
        return -1;
    }
	
    msgq = msgget(key, 0666 | IPC_CREAT);				// 권한설정 | key에 매치되는 message queue ID가 없으면 생성하는 권한
    if (msgq == -1) {
        perror("msgget()\n");
        return -1;
    }
    return msgq;
}

static int do_send(long mtype) {
    int msgq;
    key_t key;
    struct msgbuf mbuf;
    
    msgq = init_msgq();
    if(msgq == -1) {
        perror("init_msgq()\n");
        return -1;
    }
    
    memset(&mbuf, 0, sizeof(mbuf));
    mbuf.mtype = mtype;
    strncpy(mbuf.mdata, "hello world", sizeof(mbuf.mdata) - 1);
    snprintf(mbuf.mdata, sizeof(mbuf.mdata), "hello world mtype %ld", mtype);
    
    // 4번째 파라미터는 flag
    if (msgsnd(msgq, &mbuf, sizeof(mbuf.mdata), 0) == -1) {
        perror("msgsnd()\n");
        return -1;
    }
    return 0;
}

static int do_recv(long mtype) {
	int msgq;
    struct msgbuf mbuf;
    int ret;
    
    msgq = init_msgq();
    if (msgq == -1) {
        perror("init_msgq()\n");
        return -1;
    }
    
    memset(&mbuf, 0, sizeof(mbuf));
    ret = msgrcv(msgq, &mbuf, sizeof(mbuf.mdata), mtype, 0);
    if (ret == -1) {
        perror("msgrcv()\n");
        return -1;
    }
    
    printf("received msg: mtype &ld, msg [%s]\n", mbuf.mtype, mbuf.mdata);
    
    return 0;
}

int main(int argc, char** argv) {
	int ret;
    long mtype;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }
    mtype = strtol(argv[2], NULL, 10);
    if(mtype <= 0) {
        print_usage(argv[0]);
        return -1;
    }
    
    if (!strcmp(argv[1], "send")) {
        do_send();
    } else if (!strcmp(argv[1], "recv")) {
        do_recv();
    } else {
        print_usage(argv[0]);
        return -1;
    }
    
    return 0;
}
```