document.querySelectorAll(".comps-table-wrapper table").forEach(function(table) {
  var headers = table.querySelectorAll("th");
  headers.forEach(function(th, i) {
    th.addEventListener("click", function() {
      var tbody = table.querySelector("tbody");
      if (!tbody) return;
      var rows = Array.from(tbody.querySelectorAll("tr"));
      var ascending = th.getAttribute("data-sort") !== "asc";
      th.setAttribute("data-sort", ascending ? "asc" : "desc");
      rows.sort(function(a, b) {
        var ac = a.cells[i] ? a.cells[i].textContent.trim() : "";
        var bc = b.cells[i] ? b.cells[i].textContent.trim() : "";
        var an = parseFloat(ac.replace(/[^0-9.-]/g, "")) || 0;
        var bn = parseFloat(bc.replace(/[^0-9.-]/g, "")) || 0;
        if (an !== 0 || bn !== 0) return ascending ? an - bn : bn - an;
        return ascending ? (ac < bc ? -1 : 1) : (bc < ac ? -1 : 1);
      });
      rows.forEach(function(r) { tbody.appendChild(r); });
    });
  });
});
