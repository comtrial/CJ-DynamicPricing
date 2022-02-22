# cj_DynamicPricing_Project
# Dynamic Pricing

<img width="573" alt="스크린샷_2021-10-18_오후_5 27 25" src="https://user-images.githubusercontent.com/67617819/155077563-d94d6263-bd51-45c3-8b94-c6d20cd17dd6.png">

**① 주문 과제**

Dynamic Pricing 모델

상품 배송 권역 및 주소정보 등

주문 시간에 따른 유동적 운임 책정 모델

기존의 dynamic pricing 은 상품에 집중하여 상품의 종류, 무게 등을 고려한 가격 정책이나,

우리의 택배 관점에서의 dynamic pricing 점에 착안 하여 택배 관점에서의 critical 한 요인이 무엇인지에 집중
- demand의 변동성
- 특정 날에 demand가 몰림
- F / C 관점에서느 특정 날에 demand가 몰림이 곧 이후의 택배 배송에 영향을 줌(bkg - 어쩌구)

따라서 우리의 dynamic pricing 전략은 

1.  수요 / 가격 정책을 통한 demand의 평탄화 유도
2. 다양한 제약( 수요 / 가격 곡선, 평탕화의 유도 등)을 고려함에 있어 해당 제약들 마다의 알고리즘( ex greedy 알고리즘의 설명을 통한 비교) 을 세우는 것이 어려움 어필 

→ 근데 dqn 모델을 활용하면 제약식의 추가가 용이?하다.
3. 단일 가격, greedy 알고리즘 보다 더 나은 Profit의 유도

## 목차

---

1. **Overview of the Pricing Policy**
2. **ABOUT DQN**
3. **Pricing policy optimization using DQN**
4. **Policy visualization, tunning, and debugging**

## Overview of the Pricing Policy

---

### - Introductory example: Differential price response and Hi-Lo pricing

- In many cases, the development of a demand model is challenging because it has to properly capture a wide range of factors and variables that influence demand, including regular prices, discounts, marketing activities, seasonality, competitor prices, cross-product cannibalization, and halo effects.

→ 수요 모델의 challenging 한 부분은 **wide** range of factors and variable 을 고려해야 하기 때문이다.
- Once the demand model is developed, however, the optimization process for pricing decisions is relatively straightforward, and standard techniques such as linear or integer programming typically suffice

→ demand model이 결정되고 나면, 최적화 과정이 상대적으로 정직하고 표준화 된(ex) 선형 식) techniques를 가지게 된다.

### - In Our Problem

- 예측된 일주일의 demand 라는 환경 속에서 각 일의 수요량, 가격 정책을 고려한 일주일 가격 책정 정책을 r결정해야함
- 추후에 다른 제약식을 추가 할 수 있는 확장성 어필

## Greedy Algorithm

---

단일가격은 달라지는 수요에 여러 가격대입해보고 가장 큰 profit을 내는 값을 저장하고
그것이 바로 profit이됨

그리디 알고리즘의 가격 설정 방법
월요일을 제외한 나머지 가격 고정
가격을 설정한 범위 1500~ 2500원 사이의 가격을 변경해 가며(단위 50원)
7개의 가격일때의 수요를 곱해서 profit이 최대가 되는 값을 구함
그 profit이 최대가 되는 부분의 가격을 저장한다(이 가격은 월요일에 fix)
그리고 같은 방법으로 화요일을 계산한다.
7일의 가격이 결정되면 그에따른 profit을 계산해서 결과를 도출한다.

## ABOUT DQN

---

- The goal of the algorithm is to learn an action policy π that maximizes the total discounted cumulative reward (also known as the return) earned during the episode of T time steps:

![스크린샷 2021-10-28 오후 3.42.57.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c248d1af-91ac-45ef-923d-02ca1e906dc4/스크린샷_2021-10-28_오후_3.42.57.png)

- Such a policy can be defined if we know a function that estimates the expected return based on the current state and next action, under the assumption that all subsequent actions will also be taken according to the policy:

→ 현재 state 의  Action에 대한 Reward에 대한 Estimate 값을 측정

![스크린샷 2021-10-28 오후 3.59.07.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf43ea82-218b-40ce-bbbc-669f55eab325/스크린샷_2021-10-28_오후_3.59.07.png)

- Assuming that this function (known as the Q-function) is known, the policy can be straightforwardly defined as follows to maximize the return:
→ Reward를 최대화하는 action을 policy를 등록

![스크린샷 2021-10-28 오후 4.04.04.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e3d401d4-e669-46e3-a2e4-0f2c23344157/스크린샷_2021-10-28_오후_4.04.04.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1886a4a-089b-4abd-ae99-1e1628f7a321/Untitled.png)

## Pricing policy optimization using **DQN**

---

- Although the greedy algorithm we implemented above produces the optimal pricing schedule for a simple differential price-response function, **it becomes increasingly more challenging to reduce the problem to standard formulations, such as linear or integer programming, as we add more constraints or interdependencies**. In this section, we approach the problem from a different perspective and apply a generic Deep Q Network (DQN) algorithm to learn the optimal price control policy.

→ 더 많은 제약식, Demand의 변동성 등을 고려 할 경우 greedy 알고리즘으로는 한계가 존제 하며, 알고리즘을 설계하는 과정이 어려워진다.

### -Defining the environment

- At each time step t with a given state s the agent takes an action according to its
    
    $$
    policy π(s)→a
    $$
    
    and receives the reward r moving to the next state s′.
    

**Goal**

Our goal is to find a policy that prescribes a pricing action based on the current state in a way that the total profit for a selling season (episode) is maximized.

→ DQN의 목표는 현재 상태(time series)의 가격 책정을 통한 total profit(일주일 간의)의 최대화 이다.

1. First, we encode the state of the environment at any time step *t* as a vector of prices for all previous time steps concatenated with one-hot encoding of the time step itself:

    
    →  state의 정의가 해당 문제의 경우 time series 이기 때문에 step *t* 를 ont-hot encoding으로 벡터화 하여 정의한다.
    
    $$
    St=(pt−1,pt−2,…,p0,0,…) | (0,…,1,…,0)
    $$
    

1. Next, the action *a* for every time step is just an index in the array of valid price levels.

→ action 을 price level을 결정하는 index로 정의한다. 

2. Finally, the reward *r* is simply the profit of the seller.

→ reward의 경우 total time series의 종료 시점의 기업의 profit 으로 정의한다. 

## Policy visualization, tunning, and debugging

---

The learning process is very straightforward for our simplistic environment, but policy training can be much more difficult as the complexity of the environment increases. In this section, we discuss some visualization and debugging techniques that can help analyze and troubleshoot the learning process.

→ 전략(poliy)가 복잡해 질 수록 모델이 복잡해지므로 학습 과정의 시각화와 debugging 기술을 습득 할 필요가 있다. 

**1. Capturing Q-values for a given state. Click to expand the code sample.**

One of the most basic things we can do for policy debugging is to evaluate the network for a manually crafted input state and analyze the output Q-values.

→ input state에서의 Q-values를 분석하므로써 network를 평가 할 수 있다. 

We see that the network correctly suggests increasing the price (in accordance with the Hi-Lo pattern), but the distribution of Q-values if relatively flat and the optimal action is not differentiated well. If we retrain the policy with γ=0.80, then the distribution of Q-values for the same input state will be substantially different and price surges will be articulated better:

→ 감마(γ)를 낮출 수록 Q-values의 가격 책정 변화량의 변동 폭이 커진다. 

# → 이게 가지는 의미가 뭐야

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61d46733-ae05-47d0-b43a-9c11bc08fcfc/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d071d739-d3c0-4860-a5f0-2853e57d71b0/Untitled.png)

→ 감마(γ) 변화량에 따른 profit 을 확인해 보며 tunning 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06414ac5-110d-48d2-81c2-7934617dd7f5/Untitled.png)

**2. Visualization of temporal difference (TD) errors**

.In the following bar chart, we randomly selected several transitions and visualized individual terms that enter the Bellman equation:

$$
Q(s,a)=r+γmaxa′Q(s′,a′)
$$

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21e53d6f-6286-45e4-b198-c56a90190c8b/Untitled.png)

**3. visualize the correlation between Q-values and actual episode returns**

The following code snippet shows an instrumented simulation loop that records both values, and the correlation plot is shown right below (white crosses correspond to individual pairs of the Q-value and return).

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e193d512-5cf8-4845-9db5-7aa775c0afac/Untitled.png)

The correlation pattern can be much more sophisticated in more complex environments. A complicated correlation pattern might be an indication that a network fails to learn a good policy, but that is not necessarily the case (i.e., a good policy might have a complicated pattern).

→ 현재 모델의 정책(policy) 가 단순 하기 때문에 그래프가 비교적 이상적이지만, 더욱 복잡한 환경이 적용 될 경우 상관관계 또한 복잡해진다. 그래프의 복잡한 패턴은 policy를 학습하는데 실패 했다는 의미이지만 복잡한 환경에서는 부득이한 경우도 있다. 

# 내 결론

1. 제약식 알고리즘의 구현 과정의 생략 

→ 생각 보다 **존나** 큰 이점인것 같아 제약식이 아무리 많이 생겨도 막 이거를 최적화 하는 알고리즘을 안짜도 되고 그냥 제약식만 생성하면, 그 상태(t)에서의 연산만 이루어져서 total profit이 연산 되기 때문에 구현의 복잡성 크게 감소

2. 확장성

→ 위와 마찬가지의 맥락이긴 하지만 제약식의 생성에 제약이 없어
