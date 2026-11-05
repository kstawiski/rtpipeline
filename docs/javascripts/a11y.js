/* Shared with the Radioonkolog visual system (v1).
 * Names Material's search landmark for assistive technology. Stock Material
 * renders the search container as role="dialog" without a label; this only
 * adds the missing accessible name and changes no behavior or styling. */
document.addEventListener("DOMContentLoaded", () => {
  const search = document.querySelector('[data-md-component="search"]');
  if (search && !search.getAttribute("aria-label")) {
    search.setAttribute("aria-label", "Site search");
  }
});
