---
title: Thread Pool
categories:
   - MultiThreading
tags:
   - MultiThreading
---

## 개요
### 1. 스레드풀 (Thread Pool)

> 스레드 생성시에 발생하는 오버헤드를 줄이기 위해 고안된 방식
>
> - 작업큐를 이용하여 스레드가 필요한 작업들을 대기시키고
> - 일이 끝난 스레드가 작업큐에서 작업을 가져가는 방식

---
## 구현

### 1. iuclude
- Thread.h
  ```cpp
  #ifndef __THREAD_H__
  #define __THREAD_H__
  
  #include <unistd.h>
  #include <pthread.h>
  #include <vector>
  #include <iostream>
  
  class ThreadPool {
  private:
      static bool isInit;
      static std::vector<pthread_h *> threadList;
      static std::vector<void(*)()> jobQueue;
      static pthread_attr_t threadAttribute;
      static pthread_mutex_t jobMutex;
      static pthread_cond_t jobConditionVariable;
      static void *worker(void *param);
  public:
      ThreadPool();
      ThreadPool(size_t size);
      ~ThreadPool();
      bool enqueueJob(void (*job)());
      bool stop();
  }
  
  #endif	/* __THREAD_H__ */
  ```

### 2. lib
- libThread.a
  ```cpp
  #include "Thread.h"
  
  
  // static 멤버 변수의 경우에도 객체 내에서 정의 된 뒤에 객체 밖에서 따로 한번 더 정의해줘야함
  // int Test::value = 0;
  
  bool ThreadPool::isInit;
  std::vector<pthread_t *> ThreadPool::threadList;
  std::vector<void(*)()> ThreadPool::jobQueue;
  pthread_attr_t ThreadPool::threadAttribute;
  pthread_mutex_t ThreadPool::jobMutex;
  pthread_cond_t ThreadPool::jobConditionVariable;
  
  ThreadPool::ThreadPool() {
      isInit = true;
      std::cout << "[INFO] " << "Start" << std::endl;
      
      pthread_attr_init(&threadAttribute);
      pthread_attr_setdetachstate(&threadAttribute, PTHREAD_CREATE_DETACHED);
      pthread_mutex_init(&jobMutex, nullptr);
      pthread_cond_init(&jobConditionVariable, nullptr);
  
      size_t i;
      pthread_t *tmp = nullptr;
      
      for (i = 0; i < 5; ++i) {
          tmp = new pthread_t;
          threadList.push_back(tmp);
          
          pthread_create(tmp, &threadAttribute, worker, (void *)tmp);
      }
      return;
  }
  
  ThreadPool::ThreadPool(size_t size) {
      isInit = true;
      
      std::cout << "[INFO] " << "Start" << std::endl;
      
      pthread_attr_init(&threadAttribute);
      pthread_attr_setdetachstate(&threadAttribute, PTHREAD_CREATE_DETACHED);
      pthread_mutex_init(&jobMetex, nullptr);
      pthread_cond_init(&jobConditionVariable, nullptr);
      
      size_t i;
      phtread_t *tmp = nullptr;
      
      for (i = 0; i < size; ++i) {
          tmp = new pthread_t;
          threadList.push_back(tmp);
          
          pthread_create(tmp, &threadAttribute, worker, (void *)tmp);
      }
      return;
  }
  
  ThreadPool::~ThreadPool() {
      size_t i;
      for (i = threadList.size() - 1; i >= 0; --i) {
          pthread_cancel(*threadList.at(i));
          threadList.erase(threadList.begin() + i);
      }
      
      jobQueue.clear();
      threadList.clear();
      
      pthread_cond_destroy(&jobConditionVaribable);
      pthread_mutex_destroy(&jobMutex);
      pthread_attr_destroy(&threadAttribute);
      
      return;
  }
  
  bool ThreadPool::enqueueJob(void (*job)()) {
      bool retval = false;
  
      jobQueue.push_back(job);
      
      pthread_cond_signal(&jobConditionVariable);
  
      return retval;
  }
  
  bool ThreadPool::stop() {
      bool retval = false;
      
      pthread_mutex_lock(&jobMutex);
      if (isInit) {
          jobQueue.clear();
          isInit = false;
      }
      pthread_mutex_unlock(&jobMutex);
      
      return retval;
  }
  
  void &ThreadPool::worker(void *param) {
      void (*tmp)() = nullptr;
      std::cout << "[INFO] " << "Call worker" << std::endl;
      while(1==1) {
          pthread_mutex_lock(&jobMutex);
          if (jobQueue.size() == 0) {
              tmp = nullptr;
              pthread_cond_wait(&jobConditionVariable, &jobMutex);
          }
          else {
              tmp = jobQueue.at(0);
              jobQueue.erase(jobQueue.begin());
          }
          pthread_mutex_unlock(&jobMutex);
          
          if (tmp != nullptr) {
              tmp();
          }
      }
      return nullptr;
  }
  ```
  
### 3. src
- main.cpp
  ```cpp
  #include "Thread.h"
  
  void foo() {
      std::cout << "foo" << std::endl;
      sleep(3);
      return;
  }
  
  void bar(void) {
      std::cout << "bar" << std::endl;
      sleep(3);
      return;
  }
  
  int main(int argc, char** argv) {
      ThreadPool p(3);	// 3개씩 묶어서 실행할 예정
      size_t i;
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      p.enqueueJob(foo);
      std::cout << "[INFO] " << "End enqueueJob" << std::endl;
      
      sleep(30);
      return 0;
  }
  ```

