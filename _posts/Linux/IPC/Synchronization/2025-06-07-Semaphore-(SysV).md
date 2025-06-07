---
title: Semaphore (SysV)
categories:
   - IPC
tags:
   - IPC
   - Synchronization
---
## 개요

> 임계영역에 설정한 개수만큼 접근가능하도록 하는 설정
>
> - 1개만 접근 가능하도록 한다면 이건 Mutex (Mutual Exclusive)

- API

  - semget: Key를 받아 Semaphre Set(집합) 생성/열기하고 Semaphore ID 반환

  - semop: Sembuf 구조체 배열을 통해 Semaphore Set에 특정 Operation 수행
    - Semaphore의 번호 / Blocking 설정 / IPC_NOWAIT (Semaphore를 획득하지 못할때 그냥 바로 return 해서 넘어가는것)

  - semctl: Semaphore 제어
    - 주로 IPC_RMID 사용하여 Shmid 삭제