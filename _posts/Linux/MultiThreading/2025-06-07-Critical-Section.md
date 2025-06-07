---
title: Critical Section
categories:
   - MultiThreading
tags:
   - MultiThreading
---

## 임계영역 (Critical Section)

> 2개의 Thread가 같은 자원에 접근할때 **컨텍스트 스위칭**에 따라 값 변형이 제대로 적용되지 않는 문제 발생할 수 있음
>
> > - Volatile(휘발성) 쓰면?
> >   - C++에선 [[Compiler]]의 최적화만 방지함
> >   - 따라서 수정된 값을 제대로 불러온다는 보장 X
> > - Flag 변수 쓰면?
> >   - flag에 true넣는 순간 또한 컨텍스트 스위칭 발생 가능
>
> **따라서 CAS (Compare And Swap); 임계영역 사용**

### 1. 문제상황

- 컨텍스트 스위칭

  ```
  초기값 = 0; Thread1(){i++}; Thread2(){i--};
  ```

  - 각 과정은 LDR → ADD/SUB → STR 순서로 이뤄짐

    - Thread1에서 LDR과 ADD를 진행하여 값을 +1로 만들고 **컨텍스트 스위칭**

    - Thread2에서 LDR (아직 초기값 0)과 SUB/STR을 진행하여 **변수에 -1 저장**하고 컨텍스트 스위칭

    - Thread1에서 STR을 진행하여 +1을 저장

  -  **따라서 변수에 저장된 값은 +1이 됨**

- 메모리 가시성
  - 메모리에 값이 변경되기 이전에 기존에 남아있던 값을 참조할 가능성
- 코드 재배치
  - CPU Pipelining을 위해 LDR을 미리하놓거나 STR을 나중에 해서 코드가 순서대로 진행되지 않을수있음

### 2. 해결방안

> Atomic(원자적) 연산 사용

- seq_cst (임계영역): 가장 엄격
- acquire - release: acquire, release 순간은 엄격 / acquire release 사이는 느슨
- relaxed: 메모리 가시성, 코드 재배치 가능성 존재