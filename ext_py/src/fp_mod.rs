use pyo3::prelude::*;

#[pymodule]
pub mod fp_py {
    use fp::prime::{self, Binomial, Prime};
    use pyo3::exceptions::PyValueError;

    use super::*;

    fn valid_prime(p: u32) -> PyResult<prime::ValidPrime> {
        if prime::is_prime(p) {
            Ok(prime::ValidPrime::new(p))
        } else {
            Err(PyValueError::new_err(format!("{p} is not prime")))
        }
    }

    #[pyfunction]
    pub fn power_mod(p: u32, b: u32, e: u32) -> u32 {
        prime::power_mod(p, b, e)
    }

    #[pyfunction]
    pub fn log2(n: usize) -> usize {
        prime::log2(n)
    }

    #[pyfunction]
    pub fn logp(p: u32, n: u32) -> PyResult<u32> {
        Ok(prime::logp(valid_prime(p)?, n))
    }

    #[pyfunction]
    pub fn factor_pk(p: u32, n: u32) -> PyResult<(u32, u32)> {
        Ok(prime::factor_pk(valid_prime(p)?, n))
    }

    #[pyfunction]
    pub fn inverse(p: u32, k: u32) -> PyResult<u32> {
        Ok(prime::inverse(valid_prime(p)?, k))
    }

    #[pyfunction]
    pub fn minus_one_to_the_n(p: u32, i: i32) -> PyResult<u32> {
        Ok(prime::minus_one_to_the_n(valid_prime(p)?, i))
    }

    #[pyfunction]
    pub fn is_prime(p: u32) -> bool {
        prime::is_prime(p)
    }

    #[pyfunction]
    pub fn binomial(p: u32, n: u32, k: u32) -> PyResult<u32> {
        Ok(u32::binomial(valid_prime(p)?, n, k))
    }

    #[pyfunction]
    pub fn multinomial(p: u32, mut l: Vec<u32>) -> PyResult<u32> {
        Ok(u32::multinomial(valid_prime(p)?, &mut l))
    }

    #[pyfunction]
    pub fn binomial_odd_is_zero(p: u32, n: u32, k: u32) -> PyResult<bool> {
        Ok(u32::binomial_odd_is_zero(valid_prime(p)?, n, k))
    }

    #[pyfunction]
    pub fn binomial2(n: u32, k: u32) -> u32 {
        u32::binomial2(n, k)
    }

    #[pyfunction]
    pub fn multinomial2(l: Vec<u32>) -> u32 {
        u32::multinomial2(&l)
    }

    #[pyfunction]
    pub fn binomial4(n: u32, k: u32) -> u32 {
        u32::binomial4(n, k)
    }

    #[pyfunction]
    pub fn binomial4_rec(n: u32, k: u32) -> u32 {
        u32::binomial4_rec(n, k)
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("TWO", prime::TWO.as_u32())?;
        m.add("PRIMES", fp::PRIMES.to_vec())?;
        m.add("NUM_PRIMES", fp::NUM_PRIMES)?;
        m.add("PRIME_TO_INDEX_MAP", fp::PRIME_TO_INDEX_MAP.to_vec())?;
        m.add("MAX_MULTINOMIAL_LEN", fp::MAX_MULTINOMIAL_LEN)?;
        m.add("ODD_PRIMES", fp::ODD_PRIMES)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn valid_prime_conversion_stays_private() {
            let p = valid_prime(5).unwrap();
            assert_eq!(p.as_i32(), 5);
            assert_eq!(p.as_u32(), 5);
            assert_eq!(p.as_usize(), 5);
            assert_eq!(p.sum(3, 4), 2);
            assert_eq!(p.product(3, 4), 2);
            assert_eq!(p.inverse(2), 3);
            assert_eq!(p.pow(3), 125);
            assert_eq!(p.pow_mod(3, 4), 1);
            assert!(valid_prime(9).is_err());
        }

        #[test]
        fn module_helpers() {
            assert_eq!(power_mod(5, 3, 4), 1);
            assert_eq!(log2(0b1011), 3);
            assert_eq!(logp(3, 27).unwrap(), 4);
            assert_eq!(factor_pk(3, 45).unwrap(), (2, 5));
            assert_eq!(inverse(3, 2).unwrap(), 2);
            assert_eq!(minus_one_to_the_n(3, 3).unwrap(), 2);
            assert!(is_prime(7));
            assert!(!is_prime(9));
        }

        #[test]
        fn binomial_helpers() {
            assert_eq!(binomial(3, 1090, 730).unwrap(), 1);
            assert_eq!(multinomial(5, vec![1, 2, 3]).unwrap(), 0);
            assert!(binomial_odd_is_zero(3, 3, 1).unwrap());
            assert!(binomial(4, 5, 2).is_err());
            assert_eq!(binomial2(3, 1), 1);
            assert_eq!(multinomial2(vec![1, 2]), 1);
            assert_eq!(binomial4(5, 2), 2);
            assert_eq!(binomial4_rec(5, 2), 2);
        }
    }
}
