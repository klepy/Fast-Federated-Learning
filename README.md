# Fast-Federated-Learning
Implementation of the vanilla fast federated learning paper : [Fast Federated Learning by Balancing Communication Trade offs.](https://arxiv.org/abs/2105.11028)

Experiments are produced on MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

----------------------------------------------
1) Perform FL with adaptive local coefficient tau

-----------------------------------------------------------
Algorithm 1: Federated Learning with adaptive local update
-----------------------------------------------------------
1. The server broadcasts w0;
2. Workers receive and initialize w0;
3. for k= 1, ... , K do
    The server calculates tau_k using (20);
    The server broadcasts tau_k
    for j=1,...,M in parallel do
        Workers receive tau_k
        for l=1,...,tau_k do
            Workers compute g(wk-j,l)
            Workers update g(wk^j,l) as in (2);
        end
        Workers compute g(w_k^j) as in (3);
        Workers transmit g-hat(w_k) to the server;
    end
    The server receives g_hat(w_k)'s from workers;
    The server averages g_hat(w_k)'s as in (4);
    The server updates the global model as in (5);
    The server broadcast w_k+1 to workers;
   end
-----------------------------------------------------------

* Problems
- Implementation of broadcasting process in python
- How to measure wall clock time?
- 