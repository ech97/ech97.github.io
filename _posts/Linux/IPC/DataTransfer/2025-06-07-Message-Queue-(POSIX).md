---
title: Message Queue (POSIX)
categories:
   - IPC
tags:
   - IPC
   - DataTransfer
---

## 1. 개요

> Message 단위 전송

> 하나의 메세지는 하나의 덩어리처럼 처리되며 FIFO를 이용한 Message Queue
>
> - Byte stream이 아니므로 한번에 다량의 데이터 전송불가
> - Multi-reader, Multi-writer 가능
> - File I/O 기반의 동작이므로 [[File Descriptor]] 존재
>   - [[Select]], [[Epoll]] 등 I/O [[📌Multiplexing]] 가능
> - 별도의 mtype 설정없이 API 레벨에서 우선순위 적용해서 송수신
> - 메세지 올때까지 블로킹 하지 않다가, 도착하면 알아차리는 기능 제공

---
## 2. 코드

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#iuclude <sys/ipc.h>
#include <sys/msg.h>
#include <mqueue.h>

#define POSIX_MQ_NAME "/chlee"		/* 꼭 '/'가 들어가야함 */
#define MSGBUF_STR_SIZE 8192		/* mq_attr.mq_maxmsg 보다 저장할 버퍼가 커야함 (Message 단위로 들어오기때문) */

void print_usage(const char* prog) {
    printf("%s (send | recv) priority\n");
}

static int init_msgq(void) {
    // sysV에서 IPC key를 만들고 Message Queue ID를 만드는 과정은 불필요함
    
    // Message Queue Descriptor (= 일반 File Descriptor임)
    mqd_t mqd;
    
    // 4번째 인자는 attr (notification 모드를 thread/일반 설정하고, 최대 큐 개수, 최대 메세지 크기, 현재 큐 개수 저장)
    mqd = mq_open(POSIX_MQ_NAME, O_RDWR|O_CREAT, 0644, NULL);
    if (mqd == -1) {
        perror("mq_open()\n");
        return -1;
    }
    
    // NULL 넣어서 기본값으로 세팅한 attr 확인하는 과정
    struct mq_attr attr;
    memset(&attr, 0, sizeof(attr));
    if (mq_getattr(mqd, &attr) == -1) {
        perror("mq_getattr()\n");
        close(mqd);
        return -1;
    }
    
    printf("mq_flags	= %ld\n", attr.mq_flags);
    printf("mq_maxmsg	= %ld\n", attr.mq_maxmsg);
    printf("mq_msgsize	= %ld\n", attr.mq_msgsize);
    printf("mq_curmsgs	= %ld\n", attr.mq_curmsgs);
    
    return mqd;
}

static int do_send(unsigned int priority) {
    mqd_t mqd;
    char buf[MSGBUF_STR_SIZE];
    
    mqd = init_msgq();
    if (mqd == -1) {
        perror("init_msgq()\n");
        return -1;
    }
    
    memset(buf, 0, sizeof(buf));
    snprintf(buf, sizeof(buf), "hello from pid %d\n", getpid());
    if (mq_send(mqd, buf, sizeof(buf), priority) == -1) {
        perror("mq_send()\n");
        close(mqd);
        return -1;
    }
    return 0;
}

static int do_recv(unsigned int priority) {
    mqd_t mqd;
    char buf[MSGBUF_STR_SIZE];
    
    mqd = init_msgq();
    if (mqd == -1) {
        perror("init_msgq()\n");
		return -1;
    }
    
    memset(buf, 0, sizeof(buf));
    // 우선순위에 해당되는 값을 읽어들일수있으나 NULL로 설정하면 그냥 들어온 시간순으로 읽음
    if (mq_receive(mqd, buf, sizeof(buf), &priority) == -1) {
        perror("Message is too long\n");
        close(mqd);
        return -1;
    }
    printf("priority: %d, received message: %s\n", priority, buf);
    return 0;
}

int main(int argc, char** argv) {
	int ret;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }
    
    unsigned int priority;
    priority = (unsigned int) strtoul(argv[2], NULL, 10);
    
    if (!strcmp(argv[1], "send")) {
        ret = do_send(priority);
    } else if (!strcmp(argv[1], "recv")) {
        ret = do_recv(priority);
    } else {
        print_usage(argv[0]);
        return -1;
    }
    return ret;
}
```
