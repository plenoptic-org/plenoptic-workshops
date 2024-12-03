# plenoptic-workshops

> [!CAUTION]
> This repo should not be edited manually.
> 
> Every week, it will delete the documentation corresponding to pull requests that have been deleted and over-write the git history. Thus, the history will diverge with any local copies --- you should delete your local copy and re-clone this as needed.

Built workshop materials for the plenoptic package. Note that this repository
just contains the *built* documentation for the package, the source of the
documentation lies in other repos from the [plenoptic
organization](https://github.com/plenoptic-org) (generally those with "workshop"
in the title).

We use this repo essentially as a webserver: the html is placed here and then
hosted on github pages, and we treat this repo as "shallow", not tracking the
history. The building and pushing is all handled by Jenkins jobs triggered in
the main repo.
