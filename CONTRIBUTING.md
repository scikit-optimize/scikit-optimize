Contributing
============
Scikit-Optimize is an open-source project, and contributions of all kinds
are welcome. We believe in this [code of conduct](CONDUCT.md).

You can contribute documentation, examples or code, review open pull requests,
help answer questions in the forum, create visualizations, maintain project
infrastructure, write new user stories ... All efforts are equally important.


Issues
------
When opening a new issue, please follow the chosen issue template.
See [How to Report Bugs Effectively][bugs].

When describing a **bug**, please provide a [**minimal working example**][mwe],
and _full_ [traceback], if applicable, sufficient to understand
and reproduce the issue on our end.

Wrap code/verbatim/output text in [fenced code blocks][code].

[bugs]: https://www.chiark.greenend.org.uk/~sgtatham/bugs.html
[mwe]: https://en.wikipedia.org/wiki/Minimal_working_example
[traceback]: https://en.wikipedia.org/wiki/Stack_trace
[code]: https://docs.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks


Pull requests
-------------
Contributions should roughly follow the [GitHub flow] and these guidelines:

* all changes by pull request (PR);
* create a PR early while you work on
  it (good to avoid duplicated work, get broad review of functionality or API,
  or seek collaborators);
* mark your pull request as draft until it's ready for review;
* a PR should solve one problem with a minimal set of changes
  (don't mix problems together in one issue/PR);
* describe _why_ you are proposing the changes you are proposing;
* try to not rush changes (the definition of rush depends on how big your
  changes are);
* someone else has to merge your PR;
* new code needs to come with tests that cover it;
* apply [PEP8](https://www.python.org/dev/peps/pep-0008/) as much
  as possible, but not too much (`flake8` should pass);
* no merging if CI checks are red.

These are guidelines rather than hard rules to be enforced by :police_car:

All contributors are welcome, but
note that scikit-optimize is a _mature_ project, with thousands of users,
and thus a responsibility to move slow, particularly when it comes to
backwards-incompatible changes or vastly expanded functionality/scope.

**Before making a PR for a new feature, do weigh the cost-benefit
assumptions. Make sure your proposal is discussed and that it received
community support. Avoid expending effort that would go unmerged.
The [inclusion criteria for new functionality][criteria] is as strict
as that of scikit-learn.**

[GitHub flow]: https://guides.github.com/introduction/flow/
[criteria]: https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms


Jump in
-------
The development version can be installed with:

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -e .

Check out the [next milestone] for plans for the next release,
or look at some [easy issues] to get started contributing.

Additionally, running something like:

    grep -PiIR '\b(TODO|FIXME|HACK|XXX)\b' skopt

might also expose places open to some immediate improvement.

[next milestone]: https://github.com/scikit-optimize/scikit-optimize/milestones
[easy issues]: https://github.com/scikit-optimize/scikit-optimize/issues?q=is%3Aissue+is%3Aopen+label%3AEasy


Testing
-------
Run all tests by executing `pytest` in the top level directory.

To only run the subset of tests with short run time, you can use
`pytest -m 'fast_test'` (`pytest -m 'slow_test'` is also possible).
To exclude all slow running tests try `pytest -m 'not slow_test'`.

This is implemented using [pytest attributes](https://docs.pytest.org/en/latest/mark.html).
If a test runs for longer than 1 second, it should be marked as slow, otherwise as fast.


Making a release
----------------
The release procedure is almost completely automated. By tagging a new release,
CI will build all required packages and push them to PyPI. To make a release,
create a new issue and work through the following checklist:

* [ ] check if the dependencies in `setup.py` are valid or need unpinning,
* [ ] check that the `doc/whats_new/v0.X.rst` is up-to-date,
* [ ] did the last build of master succeed?
* [ ] create a [new release](https://github.com/scikit-optimize/scikit-optimize/releases),
* [ ] ping [conda-forge](https://github.com/conda-forge/scikit-optimize-feedstock).

Before making a release, we usually create a release candidate. If the next
release is v0.X, then the release candidate should be tagged v0.Xrc1.
Mark the release candidate as a "pre-release" on GitHub when you tag it.
