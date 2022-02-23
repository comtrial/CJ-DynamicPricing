# cj_DynamicPricing_Project
# Dynamic Pricing

<img width="573" alt="스크린샷_2021-10-18_오후_5 27 25" src="https://user-images.githubusercontent.com/67617819/155077563-d94d6263-bd51-45c3-8b94-c6d20cd17dd6.png">

**① 주문 과제**

Dynamic Pricing  전략 설정

1.  수요 / 가격 정책을 통한 demand의 평탄화 유도
2. 다양한 제약( 수요 / 가격 곡선, 평탕화의 유도 등)을 고려함에 있어 해당 제약들 마다의 알고리즘( ex greedy 알고리즘의 설명을 통한 비교) 을 세우는 것이 어려움 어필 
3. 단일 가격, greedy 알고리즘 보다 더 나은 Profit의 유도


**Goal**: **E-Commerce 시장의 가격 결정에 따른 시장의 수요 변화 환경에서 최적의 배송가격 결정**

- Dynamic Pricing (동적 가격 결정) 문제를 해결하기 위한 딥러닝 기반의 강화학습 모델 구축
    - 가격 결정에 따른 시장의 수요 변화 함수를 가정
    - 정해진 환경에서 7주일 동안의 최적의 가격 결정을 하는 강화학습 기반의 DQN 모델 구현
