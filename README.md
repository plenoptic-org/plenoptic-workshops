# plenoptic-workshops

> [!CAUTION]
> This repo should not be edited manually.

Built workshop materials for the plenoptic package. Note that this repository
just contains the *built* documentation for the package, the source of the
documentation lies in other repos from the [plenoptic
organization](https://github.com/plenoptic-org) (generally those with "workshop"
in the title).

We use this repo essentially as a webserver: the html is placed here and then
hosted on github pages, and we treat this repo as "shallow", not tracking the
history. The building and pushing is all handled by Jenkins jobs triggered in
the main repo.

## How to delete history every push

> [!CAUTION]
> In general, this is an exceedingly bad idea. We do this here because we don't care about our git history **at all**.

In order to keep the git history at a single commit, we remove `.git` and re-initialize it every time:

``` sh
rm -rf .git
git init
git remote add origin git@github.com:plenoptic-org/plenoptic-workshops.git
git add .
git commit -m "Updates files"
git push -f origin main
```
