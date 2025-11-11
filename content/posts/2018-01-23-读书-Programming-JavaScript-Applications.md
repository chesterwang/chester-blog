---
title: 前端系列（四） -Programming-JavaScript-Applications
date: 2018-01-23 00:01:52
tags: 
  - 读书
categories: 前端系列
---

https://www.safaribooksonline.com/library/view/programming-javascript-applications/9781491950289/ch03.html


# chapter 1 The JavaScript Revolutioin

-  [UnderScore.js](http://underscorejs.org/ ) functional programming
    - [testsuite](http://underscorejs.org/test/)
    - [annotated source code](http://underscorejs.org/docs/underscore.html)

## 1.1 anatomy of a typical modern JavaScript App
### 1.1.1 infrascture
- data store
- vpn/firewall(protect data store from unauthorized access)
- black box JSON Restful Web service Layer
- various third-party APIs,(BaaS)
- An app server/content management server to route requests and deliver pages to the client
- static content delivery network(CDN) for cached files(like iamges,javascript,CSS,and client-side templates)
- The client: Browser
![infracsturetures](http://orm-chimera-prod.s3.amazonaws.com/1234000000262/images/pjap_0101.png)

#### 1.1.1.1 NoSQL data stores

Document-oriented data stsores commonly store data in XML format, while object-oriented data stores commonly employ the JSON format.

JSON-based NoSQL data stores include MongoDB and CouchDB

#### 1.1.1.2 restful json web servies
- RESTful web services use HTTP verbs to tell the server what action the client intends. The actions supported are:
    - Create a new entry in the resource collection: HTTP POST.
    - Retrieve a resource representation: HTTP GET verb.
    - Update (replace) the resource: HTTP PUT.
    - Delete a resource: HTTP DELETE.
    - similar to  CRUD (create, retrieve, update, delete)
    - Use PUT to change the user's display name by hitting /users/userid with the updated user record. Note that this will completely replace the user record, so make sure that the representation you PUT contains everything you want it to contain.

![restful sequence](http://orm-chimera-prod.s3.amazonaws.com/1234000000262/images/pjap_0102.png)

# chapter2 functions

## 2.0 intro
### 2.0.1 programming guidelines
- don't repeat yourself
- do one thing(DOT)
- keep it simple stupid(KISS)
    - This tends to happen when a single line of code is used to accomplish more than a single atomic goal.
- less is more    

## 2.1 minimize side effects

- If your function operates on outside variables, return a copy instead of the original.
- A pure function has no side effects. It does not alter existing variables or program state in any way, and always returns the same value given the same inputs.
- Wherever possible, make sure that your functions don't change anything outside the function itself. Return amended copies rather than originals. Note that you can still alter program state. REST works this way: you get a copy of the data resource (called a representation), manipulate it, and send the copy back to the server. Sometimes performance implications make this advice impractical, but it's helpful to consider the possibility of using pure functions.

## 2.2 function definition
- It's best to avoid conditional function declarations entirely.
- Function declaration tends to encourage large piles of loosely related functions to grow in your module, with no real hints about what goes where, whether it’s public or private, or how the functions work together
- 不推荐深层次的匿名function调用,debug起来很麻烦,因为都没有名字,鼓励下面这种 method literal形式,有关系的funciton通过这种形式组织起来,grouped together
    - littering your stack trace with anonnymous functions

```js
var lightBulbAPI = {
    toggle: function () {},
    getState: function () {},
    off: function () {},
    on: function () {},
    blink: function () {}
  };

```

### 2.2.1 named function expresson
- Unlike function declarations, the name you assign is only available from within the function
- From outside the function, you must access the function through the variable it's assigned to or the parameter it's passed in on:

```js
    var lightbulbAPI = {
    toggle: function toggle() {},
    getState: function getState() {},
    off: function off() {},
    on: function on() {},
    blink: function blink() {}
  };
```
### 2.2.2 lambdas
- Lambdas are frequently confused with anonymous functions, closures, first-class functions, and higher order functions. The concepts are all similar, but they mean different things.
- It is common to confuse the words "closure" and "lambda" as synonyms. That is not accurate. Not all lambdas are closures, and not all closures are lambdas. A closure is created when a function references data that is contained outside the function scope. A lambda is a function that is used as a value (assigned to a variable or passed between functions). Some languages support lambdas but do not support closures.
- Higher-order functions are functions that consume or return functions as data. Lambdas get passed to and/or returned from higher order functions, and a function might be both a lambda and a higher order function, but not all higher order functions are lambdas.
- If a function is used as an argument or return value, it's a lambda.

### 2.2.3 Immediately Invoked Function Expressions    

```js
(function () {
  var isOn = false,
    toggle = function toggle() {
      isOn = !isOn;
      return isOn;
    },
    getState = function getState() {
      // Implementation...
    },
    off = function off() {
      // Implementation...
    },
    on = function on() {
      // Implementation...
    },
    blink = function blink() {
      // Implementation...
    },

    lightbulb = {
      toggle: toggle,
      getState: getState,
      off: off,
      on: on,
      blink: blink
    };

  test('Prototypes with IIFE.', function () {
    equal(lightbulb.toggle(), true,
      'Lightbulb turns on.');
    equal(lightbulb.toggle(), false,
      'Lightbulb turns off.');
  });
}());
```

### 2.2.4 method context
- Method invocation applies the function to the object to which it is attached. It takes the form `object.methodName() (dot notation) or object['methodName']() (square bracket notation)`
- `someMethod.call(context, argument1, argument2, ...);`
- `someMethod.apply(context, someArray);`
- `toggle = lightbulb.toggle.bind(lightbulb);`  这样 toggle代码中的this总是指向 lightbulb,如果不这样bind,那么this可能会指向更上层的object


## 2.3 function scope
- Variable scope is the section of code in which the identifier refers to the expected value. Outside a variable's scope, the variable is undefined or replaced by another variable with the same name.
- The desire to use block scope can be a good code smell that indicates that it may be time to break a function into smaller pieces in order to encourage readability, organization, and code reuse. It's a good idea to keep functions small.


### 2.3.1 hoisting

- JavaScript builds its execution environment in two passes. The declaration pass sets up the runtime environment, where it scans for all variable and function declarations and creates the identifiers. The second pass is the execution pass. After the first pass, all declared functions are available, but variables are still undefined.


```js
var x = 1;

(function () {
  console.log(x);
  var x = 2;
}());

等同于

var x = 1;

(function () {
  var x; // Declaration is hoisted and x is undefined.
  console.log(x);
  x = 2; // Initialization is still down here.
}());

结果是undefined
```

- Functions behave a little differently. Both the identifier number and the function body are hoisted,
```js
test('Function declaration hoisting', function () {
  function number() {
    return 1;
  }

  (function () {
    equal(number(), 2, 'Inner scope wins.');

    function number() {
      return 2;
    }
  }());

  equal(number(), 1, 'Outer scope still works.');
});
等价于

test('Function declaration hoisted.', function () {
  function number() {
    return 1;
  }

  (function () {
    function number() {
      return 2;
    }

    equal(number(), 2, 'Inner scope wins.');
  }());

  equal(number(), 1, 'Outer scope still works.');
});
```

- Function expressions do not share this behavior, because they do not declare a function. Instead, they declare a variable and are subject to the same variable-hoisting behavior:
- If you declare all of your variables at the top of your function, and define your functions before you try to use them, you'll never need to worry about any of this. This practice can substantially reduce scope-related bugs.

### 2.3.2 closures

In a nutshell, a closure stores function state, even after the function has returned. To create a closure, simply define a function inside another function and expose it. To expose a function, return it or pass it to another function. The inner function will have access to the variables declared in the outer function. This technique is commonly used to give objects data privacy.

## 2.4 Method Design

- some principle
    - Keep It Simple, Stupid (KISS)
    - Do One Thing (DOT), and do it well
    - Don't Repeat Yourself (DRY)
### 2.4.1 named parameters    

``` js
var newUser = createUser({
  name: 'Mike',
  showInSearch: false
});

//or 这种方法不更改其他object,创建了新的object,是的代码可重用
return $.extend({}, userProto, options);
```
### 2.4.2 Function Polymorphism

基于conditon logic branch 实现多态

This manual style of dynamic dispatch is a common technique in jQuery plug-ins in order to enable developers to add many methods to a plug-in without adding them all to the jQuery prototype (jQuery.fn). Using this technique, you can claim a single name on the jQuery prototype and add as many methods as you like to it. Users then select the method they want to invoke using:
`$(selection).yourPlugin('methodName', params);`

### 2.4.3 Generics and Collection Polymorphism
### 2.4.4  function chaining and fluent APIS

Chaining has its disadvantages. It can encourage you to do too much in a single line of code, it can encourage you to write too much procedural code, and it can be difficult to debug. It's tough to set a breakpoint in the middle of a chain.

## 2.5 Functional programming
### 2.5.1 stateless functions (pure functions)

Pure functions are stateless. This means that they do not use or modify variables, objects, or arrays that were defined outside the function. Given the same inputs, stateless functions will always return the same output. Stateless functions won't break if you call them at different times.

To maximize code reuse, try to make as many functions as possible both stateless and generic (or polymorphic). Many jQuery methods satisfy both requirements. Such functions tend to be very useful library methods.

### 2.5.2 partial application and currying
You may have heard this process described as currying. The two are commonly confused, but there is a difference. Currying is the process of transforming a function that takes multiple arguments into a chain of functions, each of which takes no more than one argument.

## 2.6  Asynchronous Operations

### 2.6.2 Promises and Deferreds

The difference between a promise and a callback is that a promise is an object that gets returned from the callee, instead of a function that gets passed into and invoked by the callee. With promises, it's much easier to add additional callbacks if you need them and to isolate those callbacks from each other so that the callback code can be organized independently of the initiating call.

# chapter 3 objects

Primitive types behave like objects when you use the property access notations, but you can't assign new properties to them. Primitives get wrapped with an object temporarily, and then that object is immediately thrown away. Any attempt to assign values to properties will seem to succeed, but subsequent attempts to access that new property will fail.

it's possible to create an object on demand at the precise moment it's needed (lazy instantiation), the singleton is reduced to an object literal.

JavaScript is not a classical OO language. It's a prototypal language.

## 3.1 classical inheritance is obsolete

Those who are unaware they are walking in darkness will never seek the light.

two foundational principles of object-oriented design:
- Program to an interface, not an implementation.
- Favor object composition over class inheritance.

Classical inheritance breaks the principle of encapsulation and tightly couples the child class to its ancestors.

- inheritance causes several problems:
    - tight coupling
    - Multiple inheritance is complicated
    - Brittle architecture
    - The gorilla/banana problem
