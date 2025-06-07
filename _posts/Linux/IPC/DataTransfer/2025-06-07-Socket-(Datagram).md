---
title: Socket (Datagram)
categories:
   - IPC
tags:
   - IPC
   - DataTransfer
---

## 개요

> Message 단위 전송

> - Message 단위로 전송되기때문에 Patial Read/Write가 없음
> - 연결에 기반을 두고있지않고 Read/Write시에 매번 주소를 넣어서 전송(Readto/Sendto)하므로 Connect API 불필요

### 1. 통신과정
#### 서버
1. Domain/Type: ```DGRAM```/Protocol에 맞는 Socket 생성
2. 소켓에 주소와 포트 (또는 파일이름) Binding
3. Server Socket을 이용하여 데이터 Sendto/Recvfrom 진행
   - 기존 Stream과 달리 Listen과 Accept의 과정이 없기때문에 Peer Socket이 없음
   - 따라서 그냥 Server Socket으로 데이터 송수신
#### 클라이언트
1. Domain/Type/Protocol에 맞는 Socket 생성
2. 소켓에 주소와 포트 (또는 파일이름) Binding
   - 기존 Stream과 달리 Server쪽에서 Peer Socket을 따로 생성하지않으므로
   - Client의 주소와 포트를 Binding해야 서로 데이터를 주고받을 수 있음
3. 기존 Stream과 달리 **주소를 담아** Sendto/Recvfrom으로 데이터 송수신

---
## 구현

```C
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#inlcude <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>			// Unix Domain 형태의 주소 (파일이름)을 sockaddr에 담기위한 구조체가 있는 헤더

#define SOCK_PATH "sock_stream_un"
#define CLIENT_SOCK_PATH "this_is_client_sock"	// Client의 주소를 Binding 하기위한 SOCK PATH
#define MAX_BUF_SIZE 128

static void print_usage(const char* prog) {
    printf("%s (server|client)\n", prog);
}

int do_server() {
	int skd;	// Socket Descriptor
    int peer;
    int ret;
    socklen_t len;
    struct sockaddr_un addr;
    struct sockaddr_un client_addr;	// 클라이언트의 주소(또는 파일이름)을 담을 구조체
    char buf[MAX_BUF_SIZE];
    
    // 1. Domain/Type/Protocol에 맞는 소켓 생성
    // Protocol을 0으로 설정하면 UNIX Domain Protocol
    // 기존 Stream에서와 달리 Type을 SOCK_DGRAM으로 설정
    skd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (skd == -1) {
        perror("skd()\n");
        return -1;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(struct sockaddr_un) - 1);
	// 소켓을 제거함으로써, 기존에 Binding 되어있는 정보지우기
    unlink(SOCK_PATH);
    
    // 2. 소켓에 주소와 포트 (또는 파일이름) Binding
    // sockaddr 포인터 형식으로 바인딩해야하며, 사이즈만 sockaddr_un으로 설정
    if (bind(skd, (struct sockaddr *)&addr, sizeof(struct sockaddr_un)) == -1) {
        perror("bind()\n");
        close(skd);
        return -1;
    }
    
    // 기존 Stream에서와 달리 Accept 대기를 위한 대기큐 설정인 Listen과 Accept가 필요없음
    
    // 3. 기존 Stream과 달리 Accept가 없기때문에 별도의 Peer Socket 존재 X
    // 따라서 Server Socket으로 오는 데이터 그대로 받아주면 됨 
    // 5-6번 파라미터의 경우 발신지의 주소와 길이를 설정해줘야하지만 이건 UNIX Domain 소켓이므로 불필요
    // 하지만 출력을 위해 일단 주소를 변수에 받음.
    len = sizeof(struct sockaddr_un);
    ret = recvfrom(skd, buf, sizeof(buf), 0,
                 (struct sock_addr*)&client_addr, &len);
    if (ret == -1) {
        perror("recvfrom()\n");
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
    struct sockaddr_un addr;			// 서버의 주소(또는 파일이름)을 담을 구조체
    struct sockaddr_un client_addr;		// 클라이언트의 주소(또는 파일이름)을 담을 구조체
    char buf[MAX_BUF_SIZE];
    
    // 1. Domain/Type/Protocol 정보를 담은 Client Socket 생성
    // Protocol을 0으로 설정하면 UNIX Domain Protocol로 설정됨
    // 기존 Stream에서와 달리 Type을 SOCK_DGRAM으로 설정
    sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (sock == -1) {
        perror("socket()\n");
        return -1;
    }

    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;
    strncpy(client_addr.sun_path, CLIENT_SOCK_PATH, sizeof(client_addr.sun_path) - 1);
    
    unlink(CLIENT_SOCK_PATH);
    // 2. Server Socket이 Message를 Recv하기 위해 Client의 주소정보 필요. 따라서 주소(또는 파일이름)를 할당하는 Binding 필요
    if (bind(sock, (struct sockaddr *)&client_addr, sizeof(struct sockaddr_un)) == -1) {
        perror("bind()\n");
        close(sock);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    memset(buf, 0, sizeof(buf));
    snprintf(buf, sizeof(buf), "this is msg from sock_stream");
    
    // 3. 기존 Stream에서 send를 썼던것과 달리 sendto를 이용하여 주소를 담아 Message send
    // 마찬가지로 addr 주소(포인터)의 형식을 sock_addr 포인터형으로 캐스팅하되, 크기는 sockaddr_un의 크기로 가야함
    ret = sendto(sock, buf, sizeof(buf), 0
                (struct sockaddr*)&addr, sizeof(struct sockaddr_un));
    if (ret < 0) {
        perror("sendto()\n");
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
