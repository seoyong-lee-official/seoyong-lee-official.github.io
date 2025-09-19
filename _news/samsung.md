---
layout: post
title: SCPC 2024
date: 2024-09-11
inline: false
related_posts: false
published: true
---

The classic **Interval Scheduling Problem** has a well-known, simple O(n) greedy solution.  
But what if the problem becomes more complex? Imagine solving *k* Interval Scheduling Problems, each with *n/k* unique independent intervals and *n* shared intervals across all *k* problems.  
Can you design an algorithm that beats the naïve O(kn) approach?

(Luckily, I managed to discover an O(nlogn) solution—drawing inspiration from **Lowest Common Ancestor (LCA)** algorithms—and implemented it just in time.)
