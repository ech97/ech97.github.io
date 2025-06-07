---
title: Move Constructor
categories:
   - Cpp
tags:
   - Cpp
---

>  이동 (복사) 생성자
### 1. 이동 (복사) 생성자를 사용하지 않는 경우

```c++
class Widget {  
private:  
    int* a;  
public:  
    Widget() : a(nullptr) { }
    Widget(int _data) : a(new int) {
	    *a = _data;
    }
    Widget(const Widget& rhs) : a(new int) {
	    *a = *(rhs.a);
    }
    ~Widget() { delete a; }
};
​  
int main() {  
	Widget A(99);
    Widget B(A);
}
```

- 아래와 같은 과정을 통해 복사가 진행됨

	1.  `A(99)`는 생성자를 통해 변수 공간을 할당받고 99 대입
	   
	2. `B(A)` 에서 복사생성자 호출
	   
	3. 객체 `A`로 부터 **값을 복사**
	   
	4. 객체 `A` 소멸자 호출

---
### 2. 이동 (복사) 생성자를 사용하는 경우

```c++
class Widget {  
private:  
    int* a;  
public:  
    Widget() : a(nullptr) { }
    Widget(int _data) : a(new int) {
	    *a = _data;
    }
    Widget(Widget&& rhs) : a(new int) {
	    a = rhs.a;
	    rhs = nullptr;
    }
    ~Widget() { delete a; }
};
​  
int main() {  
    Widget B(Widget(99));
}
```

- 아래와 같은 과정을 통해 이동 (복사)가 진행됨

	1.  `Widget(99)`는 생성자를 통해 변수 공간을 할당받고 99 대입
	   
	2. `B(Widget(99))` 에서 `Widget(99)` 가 무명/임시 객체이므로 **이동생성자** 호출
	   
	3. 객체 `Widget(99)`로 부터 **주소를 복사**
	   
	4. `Widget(99)`의 소멸자가 호출될 때 `B`로 전달한 주소 공간을 해제하지 않기 위해 `rhs = nullptr` 대입
	
	5. `Widget(99)` 임시객체의 **소멸자** 호출