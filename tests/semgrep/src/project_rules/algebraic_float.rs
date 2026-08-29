type BinaryOperation = fn(f64, f64) -> f64;
type TernaryOperation = fn(f64, f64, f64) -> f64;
type Real = f64;

pub fn forbidden_receiver_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_add(right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_sub(right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_mul(right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_div(right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_rem(right),
    ]
}

pub fn forbidden_associated_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        f64::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        f64::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        f64::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        f64::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        f64::algebraic_rem(left, right),
    ]
}

pub fn forbidden_qualified_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <f64>::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <f64>::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <f64>::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <f64>::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <f64>::algebraic_rem(left, right),
    ]
}

pub fn forbidden_type_alias_calls(left: Real, right: Real) -> [Real; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        Real::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        Real::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        Real::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        Real::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        Real::algebraic_rem(left, right),
    ]
}

pub fn forbidden_core_primitive_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        core::primitive::f64::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        core::primitive::f64::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        core::primitive::f64::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        core::primitive::f64::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        core::primitive::f64::algebraic_rem(left, right),
    ]
}

pub fn forbidden_std_primitive_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        std::primitive::f64::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        std::primitive::f64::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        std::primitive::f64::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        std::primitive::f64::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        std::primitive::f64::algebraic_rem(left, right),
    ]
}

pub fn forbidden_function_item_aliases() -> [BinaryOperation; 5] {
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let add = f64::algebraic_add;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let sub = f64::algebraic_sub;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let mul = f64::algebraic_mul;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let div = f64::algebraic_div;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let rem = f64::algebraic_rem;
    [add, sub, mul, div, rem]
}

pub fn forbidden_type_alias_function_items() -> [BinaryOperation; 5] {
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let add = Real::algebraic_add;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let sub = Real::algebraic_sub;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let mul = Real::algebraic_mul;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let div = Real::algebraic_div;
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    let rem = Real::algebraic_rem;
    [add, sub, mul, div, rem]
}

pub fn forbidden_qualified_type_alias_calls(left: Real, right: Real) -> [Real; 5] {
    [
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <Real>::algebraic_add(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <Real>::algebraic_sub(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <Real>::algebraic_mul(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <Real>::algebraic_div(left, right),
        // ruleid: mcmc.rust.no-algebraic-f64-operations
        <Real>::algebraic_rem(left, right),
    ]
}

pub const FORBIDDEN_QUALIFIED_TYPE_ALIAS_ITEMS: [BinaryOperation; 5] = [
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <Real>::algebraic_add,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <Real>::algebraic_sub,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <Real>::algebraic_mul,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <Real>::algebraic_div,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <Real>::algebraic_rem,
];

pub const FORBIDDEN_QUALIFIED_FUNCTION_ITEMS: [BinaryOperation; 5] = [
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <f64>::algebraic_add,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <f64>::algebraic_sub,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <f64>::algebraic_mul,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <f64>::algebraic_div,
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    <f64>::algebraic_rem,
];

fn accept_binary_callback(_operation: BinaryOperation) {}

pub fn forbidden_callbacks() {
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    accept_binary_callback(f64::algebraic_add);
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    accept_binary_callback(f64::algebraic_sub);
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    accept_binary_callback(f64::algebraic_mul);
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    accept_binary_callback(f64::algebraic_div);
    // ruleid: mcmc.rust.no-algebraic-f64-operations
    accept_binary_callback(f64::algebraic_rem);
}

pub fn permitted_ieee_operators(left: f64, right: f64) -> [f64; 5] {
    [
        // ok: mcmc.rust.no-algebraic-f64-operations
        left + right,
        // ok: mcmc.rust.no-algebraic-f64-operations
        left - right,
        // ok: mcmc.rust.no-algebraic-f64-operations
        left * right,
        // ok: mcmc.rust.no-algebraic-f64-operations
        left / right,
        // ok: mcmc.rust.no-algebraic-f64-operations
        left % right,
    ]
}

pub fn permitted_f32_algebraic_operations(left: f32, right: f32) -> [f32; 5] {
    [
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_add(right),
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_sub(right),
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_mul(right),
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_div(right),
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.algebraic_rem(right),
    ]
}

pub fn permitted_mul_add_forms(left: f64, right: f64, addend: f64) -> [f64; 5] {
    // ok: mcmc.rust.no-algebraic-f64-operations
    let associated_item: TernaryOperation = f64::mul_add;
    // ok: mcmc.rust.no-algebraic-f64-operations
    let qualified_item: TernaryOperation = <f64>::mul_add;
    [
        // ok: mcmc.rust.no-algebraic-f64-operations
        left.mul_add(right, addend),
        // ok: mcmc.rust.no-algebraic-f64-operations
        f64::mul_add(left, right, addend),
        // ok: mcmc.rust.no-algebraic-f64-operations
        <f64>::mul_add(left, right, addend),
        associated_item(left, right, addend),
        qualified_item(left, right, addend),
    ]
}

pub const PERMITTED_UNRELATED_FUNCTION_ITEMS: [BinaryOperation; 2] = [
    // ok: mcmc.rust.no-algebraic-f64-operations
    f64::max,
    // ok: mcmc.rust.no-algebraic-f64-operations
    <f64>::min,
];
