---
title: Move Assignment Operator
categories:
   - Cpp
tags:
   - Cpp
---

> 이동 대입 연산자: 이전에는 임시 객체의 값을 대입하는 방식이었다면 이동 대입 연산자는 rvalue reference로 할당받은 객체의 **주소를 대입**

### 1. 이동 대입 연산자를 사용하지 않는 경우

```c++
class Widget {  
private:  
    int* a;  
public:  
    Widget() : a(nullptr) { }
    Widget(int _data) : a(new int) {
	    *a = _data;
    }
    Widget& operator=(const Widget& rhs) {
        if (a != nullptr) {
            delete a;
        }
 
        a = new int;
        // 새로생긴 공간에 값 대입
        *a = *(rhs.a);
        
        // 소멸자 호출을 피할 수 없음
        return *this;
    }
    ~Widget() { delete a; }
};
​  
int main() {  
    Widget B[10];
    for (int i = 0; i < 10; ++i)
        B[i] = (Widget(i));
}
```

- 아래와 같은 과정을 통해 \[**총 2번의 할당, 1번의 해제 발생**\]

	1. `Widget(i)`를 통해 **생성자** 호출하여 메모리 공간 할당
	   
	2. `B[i] = (Widget(i))`를 통해 Widget의 **대입 연산자** 호출
	   
	3. `Widget`의 새로운 메모리 공간을 할당 (`a = new int`) 하고 대입 연산 실행
	   
	4. 값이 대입 된 `Widget`의 주소를 리턴하여 `B[i]`에 할당
	   
	5. 과정 1에서 발생한 임시객체 **소멸자** 호출

---
### 2. 이동 대입 연산자를 호출하는 경우

``` c++
class Widget {  
private:  
    int* a;  
public:  
    Widget() : a(nullptr) { }
    Widget(int _data) : a(new int) {
	    *a = _data;
    }
    Widget& operator=(Widget&& rhs) {
        if (a != nullptr) {
            delete a;
        }
 		
        /** 아래처럼 원래는 새로운 공간을 할당하고 값을 복사했으나 지금은 레퍼런스 이동 (Move)
          *a = new int
          *a = *rhs.a;
        */
        // 주소 값 복사
        a = rhs.a;
        
        // 소멸자 호출 시 이동시킨 주소를 소멸하지 못하도록 임시객체의 주소 nullptr 대입
		rhs.a = nullptr;
        return *this;
    }  
    ~Widget() { delete a; }  
};  
​  
int main() {  
    Widget B[10];  
    for (int i = 0; i < 10; ++i)   
        B[i] = (Widget(i));  
}
```

- 아래와 같은 과정을 통해 \[**총 1번의 할당, 1번의 소멸자 호출**\]

	1. `Widget(i)`를 통해 **생성자** 호출하여 메모리 공간 할당
	   
	2. `B[i] = (Widget(i))` 과정에서 `Widget(i)`은 무명 객체이므로 **이동 대입 연산자** 호출
	   
	3. 임시객체에 설정 된 주소를 `B[i]`의 **포인터 변수로 이동**
	   
	4. 소멸자 호출 시 이동 시킨 주소를 소멸하지 못하도록 임시객체의 주소에 nullptr 대입 (이미 공간을 `B[i]`로 전달시켰기때문에 nullptr 대입해도 문제없음)
	   
	5. nullptr 주소를 가진 임시객체 **소멸자** 호출