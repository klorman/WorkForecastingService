  function deleteRow(event, tableName, id) {
    event.preventDefault();
    fetch('/delete_row', {
      method: 'POST',
      body: JSON.stringify({
        table_name: tableName,
        id: id
      }),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(response => response.json()).then(data => {
      if (data.status === 'success') {
        event.target.parentNode.parentNode.remove();
        alert('Row deleted successfully!');
      } else {
        alert('Failed to delete row: ' + data.message);
      }
    });
  }