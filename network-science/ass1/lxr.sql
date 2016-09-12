select p.Id,
p.ProdName,
b.BrandName,
c1.Category1Name,
c.username,
c.country_code,
dc.city,
dc.state,
dc.country,
o.OrderId,
op.Price,
op.Discount,
cy.CurrencyName,
o.invoiceDate,
op.is_returned,
pl.on_website
from orders o
inner join order_prod op on op.OrderId = o.OrderId
inner join products p on p.Id = op.ProdId
inner join brand b on b.BrandId = p.BrandId
inner join category1 c1 on c1.Id = p.Category1
inner join currency cy on cy.CurrencyId = o.Currency
left join customers c on c.id = o.CustomerId
left join dropship_customers dc on dc.id = o.dropshipCustomerId
left join places pl on pl.customer_id = c.id
where o.invoiceDate > '2016-06-00'
and c.is_LXR_store

