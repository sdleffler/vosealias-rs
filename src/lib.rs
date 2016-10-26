//! # Walker-Vose Alias Method
//! A simple implementation of alias tables using the Walker-Vose method.

extern crate num;
extern crate rand;

use std::fmt;
use std::iter::{FromIterator, Sum};
use std::vec::Vec;

use num::{Float, NumCast, One, Zero};

use rand::Rng;
use rand::distributions::range::{Range, SampleRange};
use rand::distributions::IndependentSample;


#[derive(Debug)]
enum AliasEntry<F> {
    Aliased {
        threshold: F,
        value: usize,
        alias: usize,
    },
    Unaliased(usize),
}
use AliasEntry::*;


/// An alias table, which uses floating point probabilities of type `F` and table entries of type
/// `T`.
pub struct AliasTable<T, F> {
    table: Vec<AliasEntry<F>>,
    objs: Vec<T>,
    range: Range<usize>,
    float: Range<F>,
}


impl<T, F> fmt::Debug for AliasTable<T, F>
    where F: fmt::Debug
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "AliasTable {{ table: {:?} }}", self.table)
    }
}


impl<T, F> AliasTable<T, F>
    where F: PartialOrd + SampleRange
{
    pub fn pick<'a, R: Rng>(&'a self, rng: &mut R) -> &'a T {
        let idx = self.range.ind_sample(rng);
        let ref entry = self.table[idx];
        match entry {
            &Aliased { ref threshold, value, alias } => {
                if &self.float.ind_sample(rng) < threshold {
                    &self.objs[value]
                } else {
                    &self.objs[alias]
                }
            }
            &Unaliased(idx) => &self.objs[idx],
        }
    }
}

impl<'a, T, F: 'a> FromIterator<(T, F)> for AliasTable<T, F>
    where F: Float + NumCast + One + SampleRange + Sum<F> + Zero
{
    fn from_iter<I: IntoIterator<Item = (T, F)>>(iter: I) -> Self {
        let (objs, ps): (Vec<_>, Vec<_>) = iter.into_iter().unzip();
        let psum: F = ps.iter().cloned().sum();

        let pn = F::from(ps.len())
            .expect("Error casting usize to generic parameter F of AliasTable<T, F>");
        let pcoeff = pn / psum;

        let (mut small, mut large): (Vec<_>, Vec<_>) =
            ps.into_iter().map(|p| pcoeff * p).enumerate().partition(|&(_, p)| p < F::one());
        let mut table = Vec::new();

        while !(small.is_empty() || large.is_empty()) {
            let (l, p_l) = small.pop().unwrap();
            let (g, p_g) = large.pop().unwrap();

            table.push(Aliased {
                threshold: p_l,
                value: l,
                alias: g,
            });

            let p_g = (p_g + p_l) - F::one();

            if p_g < F::one() {
                    &mut small
                } else {
                    &mut large
                }
                .push((g, p_g));
        }

        while let Some((g, _)) = large.pop() {
            table.push(Unaliased(g));
        }

        while let Some((l, _)) = small.pop() {
            table.push(Unaliased(l));
        }

        AliasTable {
            range: Range::new(0, table.len()),
            float: Range::new(F::zero(), F::one()),
            table: table,
            objs: objs,
        }
    }
}
