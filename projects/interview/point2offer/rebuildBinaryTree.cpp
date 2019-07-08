#include <iostream>
#include <vector>
#include <map>

using namespace std;


struct tree
{
    tree(int value): value_{value}, left{nullptr}, right{nullptr} {}

    int value_;
    tree* left;
    tree* right;

};

void preTravel(tree* t) {
    if (t != nullptr){
        std::cout << t->value_ << std::endl;
        preTravel(t->left);
        preTravel(t->right);
    }
}

void inTravel(tree* t) {
    if (t != nullptr){
        inTravel(t->left);
        std::cout << t->value_ << std::endl;
        inTravel(t->right);
    }
}

void postTravel(tree* t) {
    if (t != nullptr){
        postTravel(t->left);
        postTravel(t->right);
        std::cout << t->value_ << std::endl;
    }
}


map<int, int> buildIndex(vector<int>& inorder){
    map<int, int> index;
    for (auto i{0u}; i < inorder.size(); i++){
        index[inorder[i]] = i;
    } 
    return index;
}


tree* rebuild(vector<int>& preorder, int start, int end, map<int, int>& index){
    if (start == end) {
        tree* t = new tree{preorder[start]};
        return t;
    } else if (start > end) {
        return nullptr; 
    } else {
        tree* t = new tree{preorder[start]};
        int sep = index[preorder[start]]; 
        t->left = rebuild(preorder, start+1, sep, index);
        t->right = rebuild(preorder, sep+1, end, index);
        return t;
    }
}

int main()
{
   vector<int> preorder{3, 9, 20, 15, 7}; 
   vector<int> inorder{9, 3, 15, 20, 7}; 
   int end = preorder.size() - 1;
   int start = 0; 
   auto index = buildIndex(inorder);
   tree *t = rebuild(preorder, start, end, index);
   preTravel(t);
   inTravel(t);
   postTravel(t);
}
