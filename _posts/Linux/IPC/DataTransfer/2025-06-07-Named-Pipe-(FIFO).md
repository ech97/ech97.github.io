---
title: Named Pipe (FIFO)
categories:
   - IPC
tags:
   - IPC
   - DataTransfer
---

## 개요

> Byte 단위 전송

> 서로 관련이 없는 (Unrelated) 프로세스간 통신이 가능
>
> - read 프로세스과 write 프로세스가 모두 open이어야 시작함
> - 읽을게 없는데 읽거나, 읽는게 없는데 쓰기만 하면 안되므로 모두 open하기전까진 서로 blocking

---
## 구현

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define FIFO_FILENAME "./testfifo"

static void print_usage(char *prog) {
    printf("%s (w|r)\n", prog);
	return -1;
}

static int do_reader(void) {
    int fd;
    char buf[128];
    
    fd = open(FIFO_FILENAME, O_RDONLY);
    if (fd < 0) {
        perror("open() fail\n");
        return -1;
    }
    
    read(fd, buf, sizeof(buf));	// byte stream에선 \0은 자르는 strlen보다는 sizeof 사용이 나음
    printf("writer said: %s\n", buf);
    close(fd);
    
    return 0;   
}

static int do_writer(void) {
    int fd;
    char buf[128];
    
    unlink(FIFO_FILENAME);		// 기존에 있던 파일이 있으면 지우고 다시 새로 생성
    
    if (mkfifo(FIFO_FILENAME), 0644) {	// 파일의 경로와 권한을 받아서 fifo 파일 생성
        perror("mkfifo()\n");
        return -1;
    }
    
    fd = open("FIFO_FILENAME", O_WRONLY);
    if (fd < 0) {
        perror("open() fail \n");
        return -1;
    }
    
    strncpy(buf, "hello chlee", sizeof(buf));
    write(fd, buf, strlen(buf));		// write를 통해 fifo파일에 기록

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }
    
    if (!strcmp(argv[1], "r")) {
        do_reader();
    } else if (!strcmp(argv[1], "w")) {
        do_writer();
    } else {
        print_usage(argv[0]);
        return -1;
    }
    
    return 0;
}
```
