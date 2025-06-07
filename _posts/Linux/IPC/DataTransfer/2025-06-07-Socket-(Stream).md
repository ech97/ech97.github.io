---
title: Socket (Stream)
categories:
   - IPC
tags:
   - IPC
   - DataTransfer
---

##  개요

> Byte 단위 전송
### 1. 기본 개념

> 소켓은 내부 종점 (endpoint)을 정의하기 위한 추상적인 개념
>  - 소켓간 식별을 위해 Domain과 Type에 맞는 Address 필요
#### Domain
| Domain                      | Definition | Address 정의 방법     |
| --------------------------- | ---------- | --------------------- |
| Unix domain socket          | AF_UNIX    | filepath              |
| IPv4 Internet domain socket | AF_INET    | IPv4 주소 + Port 번호 |
| IPv6 Internet domain socket | AF_INET6   | IPv6 주소 + Port 번호 |
#### Type
> Unix domain socket은 port 개수 (65535)의 제약을 받지 않으며 호스트 내의 프로세스간 통신에 사용가능

| Socket Type | Definition  | 특징                                           |
| ----------- | ----------- | ---------------------------------------------- |
| Stream      | SOCK_STREAM | Connect-orient, byte stream, reliable, 양방향  |
| Datagram    | SOCK_DGRAM  | Connectionless, unreliable (분실 우려), 양방향 |

---
### 2. 통신과정
#### 서버
1. Socket: Domain / Type: ```STREAM``` / Protocol에 맞는 Server Socket 생성
2. Bind: 특정 주소와 포트(또는 파일이름)으로 Server Socket을 Binding
3. Listen: Server Socket에 접속하는 대기큐의 크기 설정
4. Accept: Server Socket으로 연결(Connect)시도에 응답하며, Client와의 통신을 위한 Peer Socket 반환
5. Recv/Send

#### 클라이언트
1. Socket: Domain / Type / Protocol에 맞는 Client Socket 생성
   (2.) UNIX Domain이 아니라면, Client에서도 Binding을 통해 Client의 주소정보 기록함
2. Connect: Server Socket이 Binding된 주소와 포트 (또는 파일이름)을 찾아가서 연결요청
   - Client Socket과 Server Socket과의 연결이 완료되면 이후로는 Client Socket과 Peer Socket으로 통신
3. Send/Recv

---
## 구현

```c
// addr: 내 소켓에 대한 addr
// domain별로 sockaddr struct가 다름 (인터넷: IP을 담기위한 struct sockaddr, 유닉스: Filepath를 담기위한 struct sockaddr_un)
bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```

```C
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#inlcude <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>			// Unix Domain 형태의 주소 (파일이름)을 sockaddr에 담기위한 구조체가 있는 헤더

#define SOCK_PATH "sock_stream_un"
#define MAX_BUF_SIZE 128

static void print_usage(const char* prog) {
    printf("%s (server|client)\n", prog);
}

static int stream_recv(int sock, const void* buf, size_t len, int flags) {
    int read = 0;
    int ret;
    
    while(read < len) {
        // recv는 몇 Byte를 수신했는지 반환하는것을 이용하여 다 받을때까지 while문 돌도록 설정
        ret = recv(sock, (char*)buf + read, len - read, flags);
        if (ret == -1) {
            perror("recv()\n");
            return -1;
        }
        read += ret;
    }
    return 0;
}

static int stream_send(int sock, const void *buf, size_t len, int flags) {
    int written = 0;
    int ret;
    
    while(written < len) {
        ret = send(sock, (char*)buf + written, len - written, flags);
        if (ret == -1) {
            perror("send()\n");
            return -1;
        }
        written += ret;
    }
    return 0;
}

int do_server() {
	int skd;	// Socket Descriptor
    int peer;
    int ret;
    struct sockaddr_un addr;
    char buf[MAX_BUF_SIZE];
    
    // 1. Domain/Type/Protocol에 맞는 소켓 fd 생성
    // Protocol을 0으로 설정하면 UNIX Domain Protocol
    skd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (skd == -1) {
        perror("skd()\n");
        return -1;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(struct sockaddr_un) - 1);
	// 기존에 Binding 되어있는 파일 지우기
    unlink(SOCK_PATH);
    
    // 2. 소켓에 주소와 포트 (또는 파일이름) Binding
    // sockaddr 포인터 형식으로 바인딩해야하며, 사이즈만 sockaddr_un으로 설정
    if (bind(skd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) == -1) {
        perror("bind()\n");
        close(skd);
        return -1;
    }
    
    // 3. 대기큐에 5개까지 받도록 Listening
    listen(skd, 5);
    
    // 4. Client에서 Server Socket으로 접속요청(Connect)가 오면, Peer Socket의 fd 반환
    // 두세번째 파라미터는 Peer Socket의 주소, 길이 정보를 담을 버퍼인데 내부 통신 사용중이므로 NULL로 날림
    peer = accept(skd, NULL, NULL);
    if (peer == -1) {
        perror("accept()\n");
        close(skd);
        return -1;
    }
    
    // 5. Send/recv는 Peer Socket을 통해 진행 (!= Server Socket)
    ret = stream_recv(peer, buf, sizeof(buf), 0);
    if (ret == -1) {
        perror("recv()\n");
        close(skd);
        return -1;
    }
    printf("Client said: [%s]\n", buf);
    close(skd);
    return 0;
}

int do_client() {
	int ret;
    int sock;
    struct sockaddr_un addr;
    char buf[MAX_BUF_SIZE];
    
    // 1. Domain/Type/Protocol 정보를 담은 Client Socket fd 생성
    // Protocol을 0으로 설정하면 UNIX Domain Protocol로 설정됨
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("socket()\n");
        return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);
    
    // 2. Server Socket으로 Connection 요청
    // addr 주소(포인터)의 형식을 sock_addr 포인터형으로 캐스팅하되, 크기는 sockaddr_un의 크기로 가야함
    ret = connect(sock, (struct sock_addr *)&addr, sizeof(struct sockaddr_un));
    if (ret == -1) {
        perror("connect()\n");
        close(sock);
        return -1;
    }
    
    memset(buf, 0, sizeof(buf));
    snprintf(buf, sizeof(buf), "this is msg from sock_stream");
    
    // 3. Client Socket에서 Peer Socket으로 send
    ret = stream_send(sock, buf, sizeof(buf), 0);
    if (ret < 0) {
        perror("send()\n");
        close(sock);
        return -1;
    }
    close(sock);
    return 0;
}

int main(int argc, char** argv) {
	int ret;
    if (argc < 2) {
        print_usage();
        return -1;
    }
    
    if (!strncmp(argv[1] == "server")) {
        ret = do_server();
    } else if (!strncmp(argv[1] == "client")) {
        ret = do_client();
    } else {
        print_usage(argv[0]);
        return -1;
    }
    return ret;
}
```
