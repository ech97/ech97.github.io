---
title: Smart Pointer
categories:
   - Cpp
tags:
   - Cpp
---

> 사용이 끝난 포인터를 delete를 안써도 자동으로 해제
>
> - 블록을 벗어났거나
>
> - 참조하고있는 포인터가 없거나

### 1. unique_ptr

> 하나의 객체를 하나의 포인터만이 가르킬 수 있게됨
>
> - c++14부터 make_unique()를 이용하여 unique_ptr 인스턴스를 안전하게 생성가능
>   - make_unique()는 전달받은 인수로 지정된 타입의 객체를 생성하고, 생성된 객체를 가르키는 unique_ptr 반환

### 2. shared_ptr

> 특정 포인터의 복사본을 여러 객체나 코드가 가지고 있을때
>
> shared_ptr 포인터는 메모리에 대한 참조되는 숫자가 0이 될때 메모리를 해제하며
>
> 포인터가 공유될때마다 레퍼런싱 카운팅을 수행
>
> > 하나의 인스턴스를 여러곳에서 쓰면 언제 인스턴스를 소멸시켜야할지 애매해질때가있는데, 이때 shared_ptr을 이용하면 인스턴스를 참조하는 모든 곳이 사라졌을때 자동으로 소멸시켜서 유용

```cpp
#include <memory>
#include <iostream>

using namespace std;

class Knight {
    public:
        int _hp;
        int _damage;
        Knight* _target;

        Knight() {
            _hp = 10;
            _damage = 100;
            _target = NULL;
        }

        ~Knight() { cout << "Knight 소멸" << endl; }

        void Attack() {
            if (_target) {
                _target->_hp -= _damage;
            }
        }
};

// 따로 빼서 공용메모리로 관리하기 위함
class RefCountBlock {
    public:
        int _refCount;
        RefCountBlock() {
            _refCount = 1;
        }
};

template<typename T>
class SharedPtr {
    public:
        T* _ptr;
        RefCountBlock* _block;
    public:
        // 생성자
        SharedPtr() {}  // 아무것도 안하는 상태
        SharedPtr(T* ptr) : _ptr(ptr) {
            if (_ptr != NULL) {
                _block = new RefCountBlock;
                cout << "RefCount: " << _block->_refCount << endl;
            }
        }
        // 소멸자
        ~SharedPtr() {
            if (_ptr != NULL) {
                _block->_refCount -= 1;
                cout << "RefCount: " << _block->_refCount << endl;

                if (_block->_refCount == 0) {
                    delete _ptr;
                    delete _block;
                    cout << "Delete Data" << endl;
                }
            }
        }
        // 복사 생성자
        SharedPtr(const SharedPtr& sptr) : _ptr(sptr._ptr), _block(sptr._block) {
            if (_ptr != NULL) {
                _block->_refCount++;
                cout << "RefCount: " << _block->_refCount << endl;
            }
        }
        void operator = (const SharedPtr& sptr) {
            _ptr = sptr._ptr;
            _block = sptr._block;

            if (_ptr != NULL) {
                _block->_refCount;
                cout << "RecCount: " << _block->_refCount << endl;
            }
        }
};

int main(int argc, char** argv) {

    SharedPtr<Knight> k1;
    {
        SharedPtr<Knight> k2(new Knight());
        k1 = k2;
    }
    // 블록 벗어나며 k2 free됨
    k3._ptr->Attack();   // 그럼에도 k1은 use after free 발생하지않아

    return 0;
}
```

### 3. weak_ptr

> shared_ptr에서 숫자 영향 없는 버전.
>
> shared_ptr에서 서로를 가르키는 순환 구조에서는 한쪽이 해제가 안되는데 이때 weak_ptr을 쓰면 순환문제 해결가능