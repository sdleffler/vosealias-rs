//! # Walker-Vose Alias Method
//! A simple implementation of alias tables using the Walker-Vose method.

extern crate num_traits;
extern crate rand;

use std::fmt;
use std::iter::{FromIterator, Sum};
use std::vec::Vec;

use num_traits::{Float, NumCast, One, Zero};

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

/// An iterator for an alias table.
#[derive(Clone)]
pub struct AliasTableIterator<'a, T: 'a, F: 'a, R>
    where R: Rng + Sized
{
    rng: R,
    table: &'a AliasTable<T, F>
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
    /// Pick a random element from the distribution. Samples from the RNG using `ind_sample` only.
    pub fn pick<'a, R: Rng>(&'a self, rng: &mut R) -> &'a T {
        let idx = self.range.ind_sample(rng);
        let entry = &self.table[idx];
        match *entry {
            Aliased { ref threshold, value, alias } => {
                if &self.float.ind_sample(rng) < threshold {
                    &self.objs[value]
                } else {
                    &self.objs[alias]
                }
            }
            Unaliased(idx) => &self.objs[idx],
        }
    }

    /// Given an RNG, produce an iterator that picks random element from the distribution by
    /// calling `pick` repeatedly with the given RNG.
    pub fn iter<R: Rng>(&self, rng: R) -> AliasTableIterator<T, F, R> {
        AliasTableIterator {
            rng: rng,
            table: self
        }
    }
}

impl<'a, T, F: 'a> FromIterator<(T, F)> for AliasTable<T, F>
    where F: Float + NumCast + One + SampleRange + Sum<F> + Zero
{
    /// Construct an alias table from an iterator. Expects a tuple, where the left-hand element is
    /// the distribution's value, and the right-hand element is the value's weight in the distribution.
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

        table.extend(large.iter().map(|&(g, _)| Unaliased(g)));

        table.extend(small.iter().map(|&(l, _)| Unaliased(l)));

        AliasTable {
            range: Range::new(0, table.len()),
            float: Range::new(F::zero(), F::one()),
            table: table,
            objs: objs,
        }
    }
}

impl<'a, T: 'a, F, R> Iterator for AliasTableIterator<'a, T, F, R>
    where F: PartialOrd + SampleRange,
          R: Rng
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.table.pick(&mut self.rng))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (std::usize::MAX, None)
    }
}

impl<'a, T, F> IntoIterator for &'a AliasTable<T, F>
    where F: Sized + PartialOrd + SampleRange
{
    type Item = &'a T;
    type IntoIter = AliasTableIterator<'a, T, F, rand::ThreadRng>;

    /// Produces an iterator that picks random element from the distribution by calling
    /// calling `pick` repeatedly with the thread's RNG.
    fn into_iter(self) -> Self::IntoIter {
        self.iter(rand::thread_rng())
    }
}
